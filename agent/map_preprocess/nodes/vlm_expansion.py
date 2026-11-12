"""Node 7：退化区域尺寸扩展（vlm_expand_blocks）。

对 Node 6 已确认并保留的退化区域，中心不变，按尺寸档位逐步扩大观察范围，输出从基准开始连续确认仍退化的最大已测试尺寸。
semantic 与 height 两条支路独立评估、独立记录；本阶段不判断是否切换定位方式。

- 进入条件：该支路在 Node 6 已确认退化；基准范围不完整（Node 6 实际覆盖或支路所需数据足迹缺失）时不扩展，保留 Node 6 范围进入融合。
- 两图只画观察中心十字和档位框，不画道路拓扑线。
- 地图边界由程序处理：无效档位直接剔除，不画框；同心窗口下前一档无效时后面的档位也不参与；只剩基准时不调用 VLM。
- 每条支路一次请求：两张同范围的图覆盖最大有效档位，模型逐档填写观察字段；每档结论由 Node 6 的同一套 Rule 推导，
  最大尺寸由程序按档计算，停止档位之后的输出不采纳。
- 语义图用同一份整图标签；高度图与 Node 6 完全相同：DSM − H_ref（同一 100 m 范围 P20）的固定范围 Viridis 着色，
  基准区域颜色与 Node 6 一致；100 m 范围内没有有效 DSM 时停止扩展（base_not_comparable）。

扩展结果单独写入 state/degradation_extents.json 与 extensions.vlm_expansion，不改块、块尺寸或 Node 6 结论。
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from ..global_config import GlobalPreprocessConfig, PIPELINE_VERSION
from ..io import file_sha256, load_manifest, read_json, relative_path, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState
from ..utils.result_publication import protect_stage_outputs, publish_stage
from ..utils import block_features as bf
from ..utils import vlm_expansion_inputs as ei
from ..utils import vlm_review_inputs as vi
from .vlm_review import _check, _semantic_labels, color_description, height_reference, failure_fields, image_labels, image_names, run_jobs


def _meters(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def base_size(config: GlobalPreprocessConfig) -> int | float:
    return _meters(config.block_size_m)


def tier_sizes(config: GlobalPreprocessConfig) -> list:
    return [_meters(size) for size in config.vlm_expansion_sizes_m]


def _confidences(check: dict) -> dict:
    return {key: value for key, value in check.items() if key.endswith('_confidence')}


def node6_result(block: dict, kind: str) -> dict:
    """Node 6 该支路的原始结论，原样保留在每条记录中。"""
    review = block['vlm_review']['reviews'][kind]
    derived = review.get('derived') or {}
    return {'status': review['status'], 'degraded': derived.get('degraded'), 'degradation_type': derived.get('degradation_type'),
            'matched_reasons': derived.get('matched_reasons', []), 'reason_code': derived.get('reason_code'),
            'coverage': review.get('coverage')}


def base_tier(block: dict, kind: str, size: float) -> dict:
    """基准档继承 Node 6 的结论与置信度。"""
    review = block['vlm_review']['reviews'][kind]
    derived, output = review['derived'], review['output']
    check = _check(output)
    return {'size_m': size, 'state': '退化', 'source': 'node6',
            'decision_trace': derived.get('decision_trace'), 'degradation_type': derived['degradation_type'],
            'reason_code': derived['reason_code'], 'confirmed_reasons': derived['matched_reasons'],
            'evidence': check['evidence'], 'field_confidence': _confidences(check),
            'data_status': output['data_check']['status'], 'data_confidence': output['data_check']['data_confidence'],
            'uncertainty': check['uncertainty']}


def empty_tier(size: float, reason: str | None = None) -> dict:
    """未评估档位：不填造判断和置信度。"""
    from agent.vlm.block_expansion import NOT_EVALUATED
    return {'size_m': size, 'state': NOT_EVALUATED, 'source': None, 'not_evaluated_reason': reason}


def _base_result(block: dict, kind: str, config: GlobalPreprocessConfig) -> dict:
    return {'block_id': block['block_id'], 'edge_id': block['edge_id'], 'global_xy': block['global_xy'], 'review_type': kind,
            'base_size_m': base_size(config), 'node6': node6_result(block, kind)}


def skipped(block: dict, kind: str, config: GlobalPreprocessConfig, reason: str, rules: dict, detail: Any = None) -> dict:
    """跳过：max_size_m 为空，不声明基准尺寸已确认；Node 6 结果保留在 node6 字段。"""
    tiers = [empty_tier(size, reason) for size in (base_size(config), *tier_sizes(config))]
    return {**_base_result(block, kind, config), 'status': 'skipped', 'skip_reason': reason, 'skip_label': rules['skip_reasons'][reason],
            'skip_detail': detail, 'max_size_m': None, 'stop_reason': None, 'stop_label': None, 'valid_tiers_m': [], 'tiers': tiers}


def completed(block: dict, kind: str, config: GlobalPreprocessConfig, rules: dict, valid: list[float], judged: dict,
              blocked: str | None = None, extra: dict | None = None) -> dict:
    """judged：size → 单档记录（含 degraded）。程序按档计算最大尺寸，停止档位之后的记录不采纳。"""
    from agent.vlm.block_expansion import walk_tiers
    sizes = tier_sizes(config)
    best, stop, states = walk_tiers(base_size(config), sizes, valid, {k: v['degraded'] for k, v in judged.items()}, blocked)
    tiers = [base_tier(block, kind, base_size(config))]
    for size in sizes:
        state = states[size]
        tiers.append(empty_tier(size, state['not_evaluated_reason']) if state['not_evaluated_reason'] else
                     {**{k: v for k, v in judged[size].items() if k != 'degraded'}, 'state': state['state']})
    return {**_base_result(block, kind, config), 'status': 'completed', 'skip_reason': None, 'max_size_m': best, 'stop_reason': stop,
            'stop_label': rules['stop_reasons'][stop], 'valid_tiers_m': valid, 'tiers': tiers, **(extra or {})}


def base_only(block: dict, kind: str, config: GlobalPreprocessConfig, rules: dict, problem: dict) -> dict:
    """Node 6 已确认退化但基准范围数据不完整：不扩展、不调用 VLM，保留 Node 6 结论；
    Node 8 用 Node 6 实际判断的范围（coverage.image_global_bbox）作为 Patch，不声明完整 64 m 窗口。"""
    tiers = [base_tier(block, kind, base_size(config)), *(empty_tier(size, 'base_incomplete') for size in tier_sizes(config))]
    return {**_base_result(block, kind, config), 'status': 'completed', 'skip_reason': None, 'max_size_m': base_size(config),
            'stop_reason': 'base_incomplete', 'stop_label': rules['stop_reasons']['base_incomplete'], 'valid_tiers_m': [],
            'tiers': tiers, 'base_problem': problem}


def base_problem(block: dict, kind: str, window: list[int], grids: dict, image_size: tuple[int, int], resolution: float,
                 size: float) -> dict | None:
    """核对 Node 6 实际覆盖与支路数据足迹，返回不完整的说明；完整时返回 None。"""
    coverage = block['vlm_review']['reviews'][kind].get('coverage') or {}
    extent, bbox = coverage.get('image_extent_m'), coverage.get('image_global_bbox')
    if not extent or not bbox or bbox != window or min(extent) < size-resolution:
        return {'node6_coverage': coverage, 'expected_bbox': window}
    missing = ei.uncovered(grids, kind, window, image_size)
    return {'missing_sources': missing} if missing else None


def valid_tiers(block: dict, kind: str, sizes: list[float], grids: dict, image_size: tuple[int, int],
                resolution: float) -> tuple[list[float], dict]:
    """同心窗口：第一个不完整的档位之后全部剔除。"""
    valid, checks = [], {}
    for size in sizes:
        missing = ei.uncovered(grids, kind, vi.review_window(block, size, resolution), image_size)
        checks[f'{size:g}'] = missing
        if missing:
            break
        valid.append(size)
    return valid, checks


def ring_stats(missing: np.ndarray, windows: dict, outer: list[int]) -> dict:
    """各档新增环带（基准档为基准窗口）的无数据占比，本地记录并用于程序防御。"""
    scale = missing.shape[0]/(outer[2]-outer[0])
    stats, previous = {}, np.zeros(missing.shape, bool)
    for size, window in windows.items():
        a, b, c, d = (round((window[k]-outer[k % 2])*scale) for k in range(4))
        area = np.zeros(missing.shape, bool)
        area[b:d, a:c] = True
        ring = area & ~previous
        stats[f'{size:g}'] = float(missing[ring].mean()*100) if ring.any() else None
        previous = area
    return stats


def prepare_branch(manifest: dict, block: dict, kind: str, valid: list[float], labels_whole: np.ndarray,
                   config: GlobalPreprocessConfig, input_dir: Path) -> tuple[dict | None, dict]:
    """渲染一条支路的双图，返回 (job, 本地说明)；高度没有有效参考高程时 job 为 None。"""
    from PIL import Image
    resolution = manifest['resolution_m_per_px']
    size = (manifest['image']['width'], manifest['image']['height'])
    px = config.vlm_expansion_image_px
    sizes = [base_size(config), *valid]
    windows = {s: vi.review_window(block, s, resolution) for s in sizes}
    outer = windows[sizes[-1]]
    height = {}
    if kind == 'height':   # 沿用 Node 6 的着色基准和色标
        reference = height_reference(manifest, block, config)
        if reference['h_ref_m'] is None:
            return None, {'dsm_reference': reference}
        value_range = tuple(float(v) for v in config.vlm_review_dsm_range_m)
        relative = vi.read_dsm(manifest['image']['dsm'], size, outer, px)-reference['h_ref_m']
        theme = vi.dsm_color(relative, value_range)
        missing = ~np.isfinite(relative)
        height = {'dsm_reference': reference, 'dsm_color_range_m': list(value_range), 'dsm_color_legend': vi.dsm_legend(value_range)}
    else:
        labels = vi.crop_labels(labels_whole, size, outer, px)
        theme = vi.semantic_color(labels)
        missing = labels == bf.IGNORE
    span = outer[2]-outer[0]
    center = ((block['global_xy'][0]-outer[0])*px/span, (block['global_xy'][1]-outer[1])*px/span)
    frames = [(s, ei.frame_box(windows[s], outer, px)) for s in sizes]
    directory = input_dir/block['block_id']/kind
    paths = []
    for name, array in (('rgb', vi.read_rgb(manifest, outer, px)), (kind, theme)):
        image = vi.draw_center(Image.fromarray(array), center)   # 只画观察中心十字，不画道路拓扑线
        paths.append(vi.save_png(ei.draw_tier_frames(image, frames), directory/f'{name}.png'))
    review = block['vlm_review']['reviews'][kind]
    payload = {'block_id': block['block_id'], 'review_type': kind, 'base_size_m': base_size(config), 'valid_tiers_m': valid,
               'image_global_bbox': outer, 'image_size_px': [px, px], 'image_extent_m': [span*resolution]*2,
               'm_per_image_px': [span*resolution/px]*2, 'center_image_xy': list(center),
               'frames': {f'{s:g}': box for s, box in frames}, 'windows': {f'{s:g}': w for s, w in windows.items()},
               'ring_nodata_pct': ring_stats(missing, windows, outer), **height,
               'area_type': review['derived']['effective_area_type'],
               'node6': {'degradation_type': review['derived']['degradation_type'],
                         'matched_reasons': review['derived']['matched_reasons'],
                         'evidence': _check(review['output']).get('verdict', {}).get('reason')
                         or _check(review['output'])['evidence']}}
    return {'block_id': block['block_id'], 'review_type': kind, 'images': [relative_path(p) for p in paths], 'payload': payload}, {}


def model_input(payload: dict) -> dict:
    """模型只接收区域标识、双图说明、档位及继承的地面类型；Node 6 判定和精确几何留在 metadata。"""
    kind, base, valid = payload['review_type'], payload['base_size_m'], payload['valid_tiers_m']
    width, height = payload['image_extent_m']
    styles = [f'{ei.tier_style(index)[0]}={size:g} m' for index, size in enumerate((base, *valid))]
    if kind == 'semantic':
        theme = ('从整幅卫星图的自动语义分割结果裁出，底图只用预测类别着色。'
                 '各档共用同一份标签，局部错分需与 RGB 同位置核对。')
    else:
        theme = ('从 DSM 裁出，先减去与 Node 6 相同的周边低处着色基准，再用同一固定 Viridis 色带着色。'
                 '颜色表示路面、树冠、屋顶或桥面等表面的相对高度，色带基准不一定是道路高度。')
    description = (f'区域：{payload["block_id"]}；review_type：{kind}。\n'
                   '第 1 张：从原始卫星 RGB 裁出并缩放，用于辨认道路和周边地物。\n'
                   f'第 2 张：{image_names(kind)[1]}。{theme}图中不混入 RGB 底图。\n'
                   f'两张图像素位置对应，实际覆盖宽 {width:.6g} 米、高 {height:.6g} 米，对应最大有效档位。\n'
                   '两图上的中心十字、白芯黑边框和尺寸文字都由程序添加，只用于定位（图上不画道路线）；这些标记中的白色不代表专题图无数据。'
                   '各框线型：'+'，'.join(styles)+'。最外层框略向内收以便显示，实际覆盖范围以图像边缘为准。\n'
                   +color_description(kind)+'\n'
                   f'基准尺寸：{base:g} m。有效评估档位：'+'、'.join(f'{s:g} m' for s in valid)+'，从小到大。'
                   '新增环带位于本档框和上一档框之间，档位结论依据本档整窗。\n'
                   f'继承的地面类型：{payload["area_type"]}。')
    return {'block_id': payload['block_id'], 'review_type': kind, 'image_description': description}


def user_prompt(job: dict, system: dict) -> str:
    description = model_input(job['payload'])['image_description']
    return (f"请评估区域 {job['block_id']} 的 {job['review_type']} 退化是否延续到各个有效扩展档。\n\n"
            f"图像说明：\n{description}\n\n"
            f"Skill：\n{system['skill']}\n\n多尺寸 Rule：\n{system['rule']}\n\n核心 Rule：\n{system['core_rule']}\n\n"
            '按 API 指定的结构化输出格式返回 JSON，逐档先观察后判断；不要添加字段。')


def judge_tier(tier: dict, kind: str, payload: dict, system: dict, size: float) -> dict:
    """只有新增环带有有效专题数据时，当前档才有资格扩展。"""
    from agent.vlm.block_expansion import check_name, derive_tier
    ring_missing = payload['ring_nodata_pct']
    tier_key = f'{size:g}'
    empty_ring = tier_key in ring_missing and ring_missing[tier_key] in (None, 100)
    derived = derive_tier(tier, kind, payload['area_type'], system['core_rules'],
                          evidence_problem='ring_no_data' if empty_ring else None)
    check, data = tier[check_name(kind)], tier['data_check']
    return {'size_m': size, 'source': 'vlm', 'decision_trace': derived['decision_trace'], 'degraded': derived['degraded'], 'degradation_type': derived['degradation_type'],
            'reason_code': derived['reason_code'], 'confirmed_reasons': derived['matched_reasons'],
            'candidate_reasons': derived['candidate_reasons'], 'consistency_failures': derived['consistency_failures'],
            'new_ring': tier['new_ring'], 'evidence': tier['evidence'], 'window_evidence': check['evidence'],
            **({'height_comparisons': check['height_comparisons']} if 'height_comparisons' in check else {}),
            'field_confidence': _confidences(check), 'data_status': data['status'], 'data_confidence': data['data_confidence'],
            'uncertainty': check['uncertainty']}


def judge_output(output: dict, job: dict, system: dict) -> dict:
    from agent.vlm.block_expansion import tier_key
    return {size: judge_tier(output['tiers'][tier_key(size)], job['review_type'], job['payload'], system, size)
            for size in job['payload']['valid_tiers_m']}


def expand_one(job: dict, client: Any, systems: dict, config: GlobalPreprocessConfig, result_dir: Path, resume: bool = False) -> dict:
    from jsonschema import Draft202012Validator
    from agent.vlm.block_expansion import bind_expansion
    from agent.vlm.common import _generation_schema, _parse
    kind, payload = job['review_type'], job['payload']
    system = systems[kind]
    path = result_dir/f"{job['block_id']}_{kind}.json"
    schema = bind_expansion(system, job['block_id'], kind, payload['valid_tiers_m'])
    record = {'block_id': job['block_id'], 'review_type': kind, 'images': job['images'], 'valid_tiers_m': payload['valid_tiers_m'],
              'schema_version': system['rules']['schema_version'], 'pipeline_version': PIPELINE_VERSION}
    budget = config.vlm_expansion_thinking_budget
    effort = config.vlm_expansion_reasoning_effort
    thinking = {'reasoning_effort': effort} if budget != 0 and effort is not None else {}
    try:
        images = [resolve_path(p) for p in job['images']]
        prompt, labels = user_prompt(job, system), image_labels(kind)
        model = getattr(getattr(client, 'config', None), 'model', None)
        request = [system['prompt'], prompt, labels, schema, [file_sha256(p) for p in images],
                   config.vlm_expansion_max_tokens, budget, thinking, model if isinstance(model, str) else None]
        fingerprint = hashlib.sha256(json.dumps(request, ensure_ascii=False).encode('utf-8')).hexdigest()
        record['request_sha256'] = fingerprint
        if resume and path.is_file():
            try:
                prior = read_json(path)
            except (OSError, ValueError):
                prior = None
            if (isinstance(prior, dict) and prior.get('status') == 'ok'
                    and prior.get('pipeline_version') == PIPELINE_VERSION
                    and prior.get('request_sha256') == fingerprint
                    and Draft202012Validator(schema).is_valid(prior.get('output'))):
                return {**prior, 'images': job['images'], '_judged': judge_output(prior['output'], job, system)}
        result = client.infer(prompt, images=images, system=system['prompt'],
                              max_tokens=config.vlm_expansion_max_tokens, temperature=0.0, json_output=True,
                              output_schema=_generation_schema(schema), image_labels=labels,
                              enable_thinking=budget != 0, thinking_token_budget=budget or None, **thinking)
        record.update({'raw_text': result.text, 'reasoning': getattr(result, 'reasoning', ''), 'usage': result.usage,
                       'latency_seconds': round(result.latency_seconds, 2), 'inference_seconds': round(getattr(result, 'inference_seconds', 0.), 2)})
        output = _parse(result.text)
        Draft202012Validator(schema).validate(output)
        judged = judge_output(output, job, system)
        record.update({'status': 'ok', 'output': output, 'judged': {f'{k:g}': v for k, v in judged.items()}, '_judged': judged})
    except Exception as exc:  # 单条支路失败不影响其它支路；按未评估保留基准
        record.update(failure_fields(exc))
    write_json(path, {k: v for k, v in record.items() if k != '_judged'})
    return record


def render_overview(manifest: dict, config: GlobalPreprocessConfig, results: list[dict], output: Path) -> Path:
    """全图各支路最大确认尺寸：semantic 绿框、height 橙框；基准细框，扩展后粗框，基准不完整（只保留基准）的黄框。"""
    from PIL import ImageDraw
    from ..utils.global_render import read_preview
    preview, scale = read_preview(manifest['image'], config)
    image = preview.convert('RGB')
    draw = ImageDraw.Draw(image)
    side = max(image.size)
    width = max(2, side//700)
    font = vi._font(max(12, side//160))
    colors = {'semantic': (40, 220, 60), 'height': (255, 140, 0)}
    resolution = manifest['resolution_m_per_px']
    for edge in manifest['edges']:
        draw.line([(x*scale, y*scale) for x, y in edge['polyline_xy']], fill=(255, 255, 255), width=width)
    block_of = {b['block_id']: b for b in manifest['blocks']}
    for result in results:
        block = block_of[result['block_id']]
        if result['skip_reason'] == 'node6_not_confirmed':
            continue
        size = result['max_size_m'] or config.block_size_m
        box = [v*scale for v in vi.review_window(block, size, resolution)]
        expanded = (result['max_size_m'] or 0) > config.block_size_m
        draw.rectangle(box, outline=(255, 220, 0) if result['stop_reason'] == 'base_incomplete' else colors[result['review_type']],
                       width=2*width if expanded else width)
        draw.text((box[0]+3, box[1]+1), f"{result['block_id']} {result['max_size_m'] or '-'}", fill='white', font=font,
                  stroke_width=2, stroke_fill='black')
    items = [(colors['semantic'], 'semantic max confirmed size'), (colors['height'], 'height max confirmed size'),
             ((255, 220, 0), 'base only (incomplete base data)')]
    output.parent.mkdir(parents=True, exist_ok=True)
    vi.add_legend(image, items, 'Degraded extent (thick = expanded beyond base)').save(output)
    return output


def _count(results: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for result in results:
        name = str(result[key])
        counts[name] = counts.get(name, 0)+1
    return counts


def run_vlm_expansion(manifest_path: str | Path, config: GlobalPreprocessConfig, *, client: Any = None,
                      semantic_segmenter: Any = None, block_ids: list[str] | None = None, dry_run: bool | None = None,
                      resume: bool = False, logger: logging.Logger | None = None) -> dict:
    """扩展 Node 6 确认的退化区域；返回摘要（同时保存为 state/vlm_expansion.json）。块与 Node 6 结论不变。

    预览与指定块写入诊断目录。resume=True 仅复用同流程版本、同输入的成功请求。
    """
    from agent.vlm.block_expansion import load_expansion_system
    logger = logger or logging.getLogger('map_preprocess.vlm_expansion')
    dry_run = config.vlm_expansion_dry_run if dry_run is None else dry_run
    manifest_path = resolve_path(manifest_path)
    manifest = load_manifest(manifest_path)
    if not manifest.get('extensions', {}).get('vlm_review'):
        raise ValueError('需要先运行 Node 6（vlm_review）')
    if config.vlm_expansion_sizes_m[0] <= config.block_size_m:
        raise ValueError('vlm_expansion_sizes_m 必须都大于基准尺寸 block_size_m')
    task_dir = layout.task_dir_of_manifest(manifest_path)
    systems = {kind: load_expansion_system(kind) for kind in ('semantic', 'height')}
    rules = systems['semantic']['rules']
    branches = [(block, kind) for block in manifest['blocks'] if block.get('vlm_review')
                for kind in block['vlm_review']['reviews']]
    if block_ids:
        wanted = set(block_ids)
        branches = [(block, kind) for block, kind in branches if block['block_id'] in wanted]
    diagnostic = dry_run or block_ids is not None
    if diagnostic:
        input_dir, result_dir = layout.diagnostic_dirs(task_dir, 'node7', dry_run=dry_run)
    else:
        input_dir = layout.figure(task_dir, 'vlm_expansion_inputs')
        result_dir = layout.state_dir(task_dir)/'vlm_expansion'
    summary_path = (result_dir if diagnostic else layout.state_dir(task_dir))/'vlm_expansion.json'
    overview_path = input_dir/'overview.png' if diagnostic else layout.figure(task_dir, 'vlm_expansion')
    extents_path = summary_path.with_name('degradation_extents.json')
    outputs = (input_dir, result_dir, summary_path, extents_path, overview_path)
    publication = nullcontext() if diagnostic else protect_stage_outputs(
        manifest_path, outputs, keep=(result_dir,) if resume else ())
    with publication:
        for directory in (input_dir, *(() if resume else (result_dir,))):
            if directory.exists():
                shutil.rmtree(directory)
        result_dir.mkdir(parents=True, exist_ok=True)

        resolution = manifest['resolution_m_per_px']
        image_size = (manifest['image']['width'], manifest['image']['height'])
        confirmed = [(b, k) for b, k in branches if k in b['vlm_review']['scene_types']]
        results: dict[tuple[str, str], dict] = {(b['block_id'], k): skipped(b, k, config, 'node6_not_confirmed', rules)
                                                for b, k in branches if k not in b['vlm_review']['scene_types']}
        jobs = []
        if confirmed:
            labels = _semantic_labels(manifest, config, semantic_segmenter, logger)
            grids = ei.coverage_grids(manifest, labels, any(k == 'height' for _, k in confirmed))
        sizes = tier_sizes(config)
        pending: dict[tuple[str, str], tuple[dict, list[float], dict]] = {}
        for block, kind in confirmed:
            key = (block['block_id'], kind)
            window = vi.review_window(block, config.block_size_m, resolution)
            problem = base_problem(block, kind, window, grids, image_size, resolution, config.block_size_m)
            if problem:
                results[key] = base_only(block, kind, config, rules, problem)
                continue
            valid, checks = valid_tiers(block, kind, sizes, grids, image_size, resolution)
            extra = {'tier_coverage': checks}
            if not valid:  # 只有完整基准有效：不调用 VLM
                results[key] = completed(block, kind, config, rules, valid, {}, extra=extra)
                continue
            job, info = prepare_branch(manifest, block, kind, valid, labels, config, input_dir)
            if job is None:  # 100 m 范围内没有有效 DSM：停止扩展，保留基准结论
                results[key] = completed(block, kind, config, rules, valid, {}, 'base_not_comparable', {**extra, **info})
                continue
            directory = input_dir/block['block_id']
            write_json(directory/f'{kind}_input.json', model_input(job['payload']))
            write_json(directory/f'{kind}_metadata.json', {**job['payload'], 'tier_coverage': checks})
            jobs.append(job)
            pending[key] = (block, valid, extra)
        logger.info('Node 7：%d 条支路，已确认 %d，VLM 任务 %d', len(branches), len(confirmed), len(jobs))
        summary = {'status': 'dry_run' if dry_run else 'completed', 'schema_version': rules['schema_version'],
                   'base_size_m': base_size(config), 'tier_sizes_m': sizes, 'branch_count': len(branches), 'job_count': len(jobs),
                   'jobs_by_type': {k: sum(j['review_type'] == k for j in jobs) for k in systems},
                   'inputs_dir': relative_path(input_dir), 'results_dir': relative_path(result_dir), 'path': relative_path(summary_path)}
        if dry_run:
            for job in jobs:
                kind = job['review_type']
                images = ''.join(f'{label}\n<image {p}>\n' for label, p in zip(image_labels(kind), job['images']))
                (result_dir/f"{job['block_id']}_{kind}_prompt.txt").write_text(
                    f"[system]\n{systems[kind]['prompt']}\n\n[user]\n{images}{user_prompt(job, systems[kind])}\n", encoding='utf-8')
            write_json(summary_path, summary)
            return summary

        records: list[dict] = []
        if jobs:
            owned = client is None
            if owned:
                from agent.vlm.client import VLMClient
                client = VLMClient()
            try:
                records = run_jobs(lambda job: expand_one(job, client, systems, config, result_dir, resume), jobs,
                                   config.vlm_expansion_workers, 'Node 7 VLM 扩展', logger)
            finally:
                if owned:
                    client.close()
        for record in records:
            key = (record['block_id'], record['review_type'])
            block, valid, extra = pending[key]
            blocked = None if record['status'] == 'ok' else 'request_failed'
            results[key] = completed(block, record['review_type'], config, rules, valid, record.get('_judged', {}), blocked, extra)
        ordered = [results[(b['block_id'], k)] for b, k in branches]
        overview = render_overview(manifest, config, ordered, overview_path)
        finished = [r for r in ordered if r['status'] == 'completed']
        summary.update({'completed_count': len(finished), 'skipped': _count([r for r in ordered if r['status'] == 'skipped'], 'skip_reason'),
                        'max_size_counts': _count(finished, 'max_size_m'), 'stop_reason_counts': _count(finished, 'stop_reason'),
                        'errors': [{'block_id': r['block_id'], 'review_type': r['review_type'], 'error': r['error']}
                                   for r in records if r['status'] == 'error'],
                        'degradation_extents': relative_path(extents_path), 'overview': relative_path(overview)})
        write_json(extents_path, {'task_id': manifest.get('task_id'), 'image': manifest['image']['path'],
                                  'resolution_m_per_px': resolution, 'base_size_m': base_size(config), 'tier_sizes_m': sizes,
                                  'meaning': rules['program_mapping']['meaning'], 'count': len(ordered), 'extents': ordered})
        write_json(summary_path, summary)
        if diagnostic:
            return summary
        manifest.setdefault('extensions', {})['vlm_expansion'] = summary
        manifest['files'].update({'degradation_extents': relative_path(extents_path), 'vlm_expansion_overview': relative_path(overview),
                                  'vlm_expansion': relative_path(summary_path)})
        publish_stage(manifest_path, manifest, 'vlm_expansion')
        logger.info('Node 7 完成：完成 %d，跳过 %s，最大尺寸 %s，失败 %d', len(finished), summary['skipped'], summary['max_size_counts'],
                    len(summary['errors']))
        return summary


def vlm_expand_blocks(state: PreprocessState, *, config: GlobalPreprocessConfig, client: Any = None,
                      semantic_segmenter: Any = None) -> dict:
    if (state.get('vlm_review') or {}).get('status') == 'dry_run':
        return {'vlm_expansion': {'status': 'skipped', 'reason': 'node6_dry_run'}}
    summary = run_vlm_expansion(state['manifest_path'], config, client=client, semantic_segmenter=semantic_segmenter,
                                logger=logging.getLogger(f"map_preprocess.{state['task_id']}"))
    return {'vlm_expansion': summary}
