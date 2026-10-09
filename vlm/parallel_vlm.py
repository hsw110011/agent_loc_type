"""并行 VLM worker：消费 bbox 可视化图与区域 JSON，输出每个区域的独立结果。"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .client import VLMClient, VLMError
from jsonschema import ValidationError
from .common import _parse, _generation_schema, read_config, resolve_path, write_json


class ParallelVLMError(RuntimeError):
    """并行 VLM 输入、输出或推理失败。"""


def _write_json(path: Path, value: Any) -> None:
    write_json(path, value)


def _load_job(job: dict[str, Any]) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    if not isinstance(job, dict):
        raise ParallelVLMError("parallel job 必须为对象")
    area_id = job.get("area_id")
    if not isinstance(job.get('region_json_path'), str) or not job['region_json_path']:
        raise ParallelVLMError('parallel job 缺少 region_json_path')
    region_json_path = resolve_path(job.get("region_json_path", ""))
    if not isinstance(area_id, str) or not area_id:
        raise ParallelVLMError("parallel job 缺少 area_id")
    if not region_json_path.is_file():
        raise ParallelVLMError(f"{area_id} 的统一 regions.json 输入不存在")
    try:
        regions = json.loads(region_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ParallelVLMError(f"无法读取统一区域 JSON: {region_json_path}") from exc
    if not isinstance(regions, dict) or not isinstance(regions.get('areas'), list):
        raise ParallelVLMError('regions.json 必须包含 areas 数组')
    matches = [area for area in regions['areas'] if isinstance(area, dict) and area.get('area_id') == area_id]
    if len(matches) != 1:
        raise ParallelVLMError(f'{area_id} 在 regions.json 中必须且只能出现一次')
    area = matches[0]
    roi_fields = ('roi_image_path', 'roi_crop_bbox', 'roi_polygon_local', 'roi_offset_xy', 'roi_outside_region')
    if any(key not in area for key in roi_fields):
        raise ParallelVLMError(f'{area_id} 缺少 ROI 路径或坐标说明')
    payload = {key: value for key, value in regions.items() if key not in {'areas', 'legend'}}
    payload.update({'area_id': area_id, 'area': area,
                    **{key: area[key] for key in roi_fields}})
    # 区域 JSON 保存的路径是加载依据；清单只负责指定本次要加载的区域 JSON。
    paths = []
    for key in ('annotated_image_path', 'original_image_path', 'roi_image_path'):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ParallelVLMError(f"{area_id} 的区域 JSON 缺少 {key}")
        path = resolve_path(value)
        if not path.is_file():
            raise ParallelVLMError(f"{area_id} 的 {key} 不存在: {path}")
        if job.get(key) and resolve_path(job[key]) != path:
            raise ParallelVLMError(f"{area_id} 的清单与区域 JSON 路径不一致: {key}")
        paths.append(path)
    annotated, original, roi = paths
    return payload, region_json_path, annotated, original, roi


def _prompt(payload: dict[str, Any]) -> str:
    area = payload["area"]
    return ("你是并行区域复核 VLM。只复核 JSON 指定的 area_id，不重新编号，不扩大或缩小 bbox。"
            "第一张是带编号的全局框图，第二张是原始全图，第三张是当前区域的原始 ROI 裁剪。先按 area_id 定位，再结合全局上下文与 ROI 细节判断。"
            "bbox 为完整图像上的四个顺时针顶点，可表示倾斜矩形或凸四边形。ROI 中区域外为透明或黑色留白，不是真实地物；只分析区域内部。"
            "ROI 像素坐标加 roi_offset_xy 才是完整图像坐标，不把 bbox 第一个顶点当作 ROI 左上角。"
            "彩色框、标签和 JSON 都是程序提供的任务输入，不是卫星图中的地物。"
            "请返回对该区域的可见证据、风险判断、不确定性和需要补充的信息；不能从不可见信息推断。\n\n"
            f"区域输入 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            f"当前 area_id={area['area_id']}，bbox 使用原图像素坐标。")


def run_parallel_vlm(job: dict[str, Any], client: VLMClient | None = None, *, max_tokens: int | None = None, config: dict | None = None) -> dict[str, Any]:
    if client is None:
        with VLMClient() as shared:
            return run_parallel_vlm(job, shared, max_tokens=max_tokens, config=config)
    payload, region_json_path, annotated, original, roi = _load_job(job)
    vlm = client or VLMClient()
    config = config or load_assessment_config()
    result = vlm.infer(_prompt(payload) + "\nSkill：\n" + config['skill'] +
                       "\nRule：\n" + config['rule'] + "\n输出 schema：\n" + json.dumps(config['schema'], ensure_ascii=False),
                       images=[annotated, original, roi], system=config['prompt'],
                       max_tokens=max_tokens, temperature=0.0, json_output=True,
                       output_schema=_generation_schema(config['schema']))
    assessment = _parse(result.text)
    from jsonschema import Draft202012Validator
    Draft202012Validator(config['schema']).validate(assessment)
    if assessment.get('area_id') != payload['area_id']:
        raise ParallelVLMError('评估输出必须保留输入 area_id')
    output_path = region_json_path.parent.parent / 'node5' / (payload['area_id'] + '.result.json')
    output = {
        "area_id": payload["area_id"],
        "task_id": payload.get("task_id"),
        "input": {
            "region_json_path": str(region_json_path),
            "annotated_image_path": str(annotated),
            "original_image_path": str(original),
            "roi_image_path": str(roi),
        },
        "assessment": assessment,
        "usage": result.usage,
        "latency_seconds": result.latency_seconds,
        "request_id": result.request_id,
        "finish_reason": result.finish_reason,
    }
    _write_json(output_path, output)
    return {**output, "result_json_path": str(output_path)}


def run_parallel_jobs(jobs: list[dict[str, Any]], *, client: VLMClient | None = None,
                      max_workers: int | None = None, max_tokens: int | None = None,
                      config: dict | None = None) -> dict[str, Any]:
    if max_workers is not None and (isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0):
        raise ValueError("parallel_workers 必须为正整数")
    if not isinstance(jobs, list):
        raise ParallelVLMError("parallel_inputs 必须为列表")
    if not jobs:
        return {"dispatch_status": "no_candidate_jobs", "results": [], "errors": []}
    config = config or load_assessment_config()
    workers = max_workers or min(4, len(jobs))
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("parallel_workers 必须为正整数")
    results, errors = [], []
    if client is None:
        with VLMClient() as shared:
            return run_parallel_jobs(jobs, client=shared, max_workers=workers,
                                     max_tokens=max_tokens, config=config)
    # 线程复用同一客户端、启动锁与请求信号量，只加载一份 VLM。
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {pool.submit(run_parallel_vlm, job, client, max_tokens=max_tokens, config=config): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except (OSError, ValueError, VLMError, ParallelVLMError, ValidationError) as exc:
                errors.append({"area_id": job.get("area_id"), "error": str(exc)})
    results.sort(key=lambda item: item["area_id"])
    errors.sort(key=lambda item: str(item.get("area_id")))
    status = "completed" if not errors else ("partial_failure" if results else "failed")
    return {"dispatch_status": status, "results": results, "errors": errors}


def assess_regions(state: dict[str, Any], *, client: VLMClient | None = None,
                      max_workers: int | None = None, max_tokens: int | None = None) -> dict[str, Any]:
    """可直接注册为下游节点；输入为 prepare_parallel_inputs 生成的状态。"""
    dispatch = run_parallel_jobs(state.get("parallel_inputs", []), client=client,
                                  max_workers=state.get('parallel_workers') if max_workers is None else max_workers,
                                  max_tokens=state.get('parallel_max_tokens') if max_tokens is None else max_tokens)
    _write_json(resolve_path(state['trace_dir']) / 'node5/assessments.json', dispatch)
    return {"parallel_vlm": dispatch}


def load_assessment_config(config_dir: str | Path | None = None) -> dict:
    """业务规则尚未填写时阻止推理；逐定位方式规则一并注入。"""
    root = resolve_path(config_dir if config_dir else Path(__file__).parent / 'loc_type_sys')
    from jsonschema import Draft202012Validator
    schema = json.loads(read_config(root / 'schema.json'))
    Draft202012Validator.check_schema(schema)
    rules = [read_config(root / 'rule.yaml')]
    rules.extend(read_config(path) for path in sorted((root / 'rule_loc_type').glob('*.yaml')))
    return {'prompt': read_config(root / 'prompt.md'), 'skill': read_config(root / 'skill.md'),
            'rule': '\n\n'.join(rules), 'schema': schema}
