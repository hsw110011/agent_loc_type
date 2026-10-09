"""Node 3：单张道路增强图的候选识别，使用本机 API Structured Outputs。"""
from __future__ import annotations

import base64
import copy
import io
import json
import time
import yaml
from pathlib import Path
from typing import TypedDict

from .client import VLMClient
from PIL import Image
from jsonschema import Draft202012Validator
from .common import PositioningError, _digest, _parse, read_config, record_trace, validate_region, regions_overlap, region_vertices, write_json, resolve_path, relative_paths


class CandidateState(TypedDict, total=False):
    path_base: str
    image_path: str
    primary_mode: str
    image_id: str
    acquisition_date: str | None
    gsd_m: float | None
    task_context: dict
    max_regions: int | None
    image: dict
    input_view: dict
    image_digest: str
    encoded_image: str
    model_candidates: dict
    inference: dict
    result: dict
    output_dir: str
    task_id: str
    artifacts: dict
    parallel_inputs: list[dict]
    parallel_vlm: dict
    run_parallel_vlm: bool
    parallel_workers: int | None
    parallel_max_tokens: int | None
    road_mask_path: str
    road_encoded_image: str
    road_extraction: dict
    geo_coordinates: dict | None
    source_image_path: str
    source_image_digest: str
    rgb_image_path: str
    raster_metadata: dict
    coordinate_transform: dict
    coordinate_transform_path: str
    trace_dir: str
    trace_json_path: str
    trace_txt_path: str
    max_output_pixels: int
    pathfinder_checkpoint: str
    pathfinder_device: str
    pathfinder_threads: int
    tile_size: int
    tile_overlap: int
    road_mask_png_path: str
    road_enhanced_path: str
    road_boundary_path: str
    road_polygons_path: str
    road_polygons_geojson_path: str | None

    candidate_view: dict
    runtime_dir: str
    candidate_json_path: str
    candidate_original_json_path: str
    region_json_path: str


def load_candidate_config(config_dir: str | Path | None = None) -> dict:
    """顺序读取 Prompt → Skill → Rule → Schema，不启动模型。"""
    root = resolve_path(config_dir if config_dir else Path(__file__).parent / 'candidate_area_sys')
    files = [('prompt', 'prompt.md'), ('skill', 'skill.md'), ('rule', 'rule.yaml'), ('schema', 'schema.json')]
    texts = {key: read_config(root / filename) for key, filename in files}
    schema = json.loads(texts['schema'])
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise PositioningError(f'候选 schema 定义不合法: {exc}') from exc
    if set(schema.get('properties', {})) != {'areas'}:
        raise PositioningError('Node 3 schema 顶层只允许 areas；请同步节点契约后再修改字段')
    area = schema['properties']['areas']['items']
    fields = {'area_id', 'bbox', 'scene_description', 'selection_reason', 'matched_rules'}
    if set(area['properties']) != fields or set(area['required']) != fields:
        raise PositioningError('Node 3 区域契约必须包含 area_id/bbox/scene_description/selection_reason/matched_rules')
    rules = yaml.safe_load(texts['rule'])
    catalog = {}
    for item in [*rules.get('candidate_patterns', []), rules.get('selection_policy', {})]:
        identifier, name = item.get('id'), item.get('title')
        if not isinstance(identifier, str) or not identifier.strip() or not isinstance(name, str) or not name.strip():
            raise PositioningError('候选规则必须提供非空 id 与 title')
        if identifier in catalog:
            raise PositioningError(f'重复 rule_id: {identifier}')
        catalog[identifier] = name
    # 从 Rule 编译 ID/名称成对约束，避免维护两份枚举或让模型编造规则。
    area['properties']['matched_rules']['items'] = {'anyOf': [
        {'type': 'object', 'additionalProperties': False, 'required': ['rule_id', 'rule_name'],
         'properties': {'rule_id': {'type': 'string', 'enum': [identifier]},
                        'rule_name': {'type': 'string', 'enum': [name]}}}
        for identifier, name in catalog.items()]}
    Draft202012Validator.check_schema(schema)
    return {**texts, 'schema': schema, 'files': files, 'root': root, 'rule_catalog': catalog, 'rules': rules}


def validate_candidates(data: dict, schema: dict, width: int, height: int,
                        max_regions: int | None = None, *, check_overlap: bool = True) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: str(list(e.absolute_path)))
    if errors:
        error = errors[0]
        raise PositioningError(f'候选 schema 校验失败 {list(error.absolute_path)}: {error.message}')
    if max_regions is not None and len(data['areas']) > max_regions:
        raise PositioningError('超过显式候选数量上限，不自动截断')
    previous = []
    geometry_errors = []
    for index, area in enumerate(data['areas'], 1):
        if area['area_id'] != f'Area_{index}':
            raise PositioningError('候选 ID 必须按 Area_1、Area_2…顺序编号，不重排或重新编号')
        try:
            validate_region(area['bbox'], width, height)
        except PositioningError as exc:
            geometry_errors.append(f"{area['area_id']} 的 bbox={area['bbox']} 无效：{exc}")
            continue
        for other in previous:
            if check_overlap and regions_overlap(area['bbox'], other['bbox']):
                geometry_errors.append(f"{other['area_id']} 与 {area['area_id']} 的 bbox 存在面积重叠；请缩小、拆分或合并区域，边界接触允许")
        previous.append(area)
        if not area['scene_description'].strip() or not area['selection_reason'].strip():
            raise PositioningError('场景描述和选择理由不能只包含空白')
        rule_ids = [rule['rule_id'] for rule in area.get('matched_rules', [])]
        if len(rule_ids) != len(set(rule_ids)):
            raise PositioningError(f"{area['area_id']} 重复引用 rule_id")
    if geometry_errors:
        raise PositioningError('\n'.join(geometry_errors))


def select_disjoint_candidates(data: dict) -> tuple[dict, dict]:
    """保持模型发现顺序，排除与已保留区域相交的候选；不改变任何顶点或规则依据。"""
    kept, dropped, mapping = [], [], []
    for area in data['areas']:
        conflicts = [item['source_area_id'] for item in mapping
                     if regions_overlap(area['bbox'], kept[item['index']]['bbox'])]
        if conflicts:
            dropped.append({'source_area_id': area['area_id'], 'conflicts_with': conflicts,
                            'bbox': area['bbox'], 'matched_rules': area['matched_rules']})
            continue
        result = copy.deepcopy(area)
        result['area_id'] = f'Area_{len(kept) + 1}'
        mapping.append({'source_area_id': area['area_id'], 'area_id': result['area_id'], 'index': len(kept)})
        kept.append(result)
    return {'areas': kept}, {'policy': 'keep_first_in_model_search_order', 'dropped': dropped,
                            'id_mapping': [{k: v for k, v in item.items() if k != 'index'} for item in mapping]}


def identify_candidates(state: CandidateState, *, client: VLMClient | None = None,
                        max_tokens: int = 4096, config_dir: str | Path | None = None) -> dict:
    """读取配置、保存实际输入、首次请求时加载 VLM、校验并映射 bbox。"""
    if client is None:
        with VLMClient() as shared:
            return identify_candidates(state, client=shared, max_tokens=max_tokens, config_dir=config_dir)
    started = time.monotonic()
    directory = Path(__file__).parent / 'runtime' / state['task_id'] / 'node3'
    directory.mkdir(parents=True, exist_ok=False)
    try:
        config = load_candidate_config(config_dir)
        limit = state.get('max_regions')
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError('max_regions 必须为空或正整数')
        if _digest(state['image_path']) != state['image_digest']:
            raise PositioningError('处理后的 TIFF 已改变')
        enhanced_path = resolve_path(state['road_enhanced_path'])
        enhanced_digest = _digest(str(enhanced_path))
        encoded = client._prepare_image(enhanced_path)
        input_path = directory / 'vlm_input.png'
        input_path.write_bytes(base64.b64decode(encoded.split(',', 1)[1]))
        with Image.open(io.BytesIO(input_path.read_bytes())) as image:
            width, height = image.size
        original_width, original_height = state['image']['width'], state['image']['height']
        with Image.open(enhanced_path) as original:
            if original.size != (original_width, original_height):
                raise PositioningError('道路增强图与处理后的 TIFF 尺寸不一致')
        view = {'image_path': str(input_path.resolve()), 'width': width, 'height': height,
                'coordinate_system': 'vlm_input_pixel_edges_clockwise_quadrilateral',
                'original_width': original_width, 'original_height': original_height,
                'scale_to_original': [original_width / width, original_height / height],
                'source_enhanced_path': str(enhanced_path), 'source_sha256': enhanced_digest,
                'note': '坐标基于保存的实际 API 输入图，不是原图或模型内部 patch 坐标。'}
        schema = copy.deepcopy(config['schema'])
        bbox = schema['properties']['areas']['items']['properties']['bbox']
        bbox['items']['prefixItems'] = [{'type': 'number', 'minimum': 0, 'maximum': bound}
                                        for bound in (width, height)]
        bbox['items']['items'] = False
        if limit is not None:
            schema['properties']['areas']['maxItems'] = limit
        snapshot = directory / 'prompt_system'
        snapshot.mkdir()
        metadata = []
        for key, filename in config['files']:
            source = config['root'] / filename
            (snapshot / filename).write_text(source.read_text(), encoding='utf-8')
            metadata.append({'order': len(metadata) + 1, 'file': str(source), 'sha256': _digest(str(source)),
                             'api_location': 'response_format.json_schema.schema' if key == 'schema' else 'messages[0].content'})
        context = {'input_view': view, 'prompt_loading': metadata, 'effective_schema': schema}
        write_json(directory / 'context.json', context)
        system = config['prompt'] + '\n\n## Skill\n' + config['skill'] + '\n\n## Rule\n' + config['rule']
        prompt = (f'仅分析这一张道路增强卫星图，实际输入尺寸为 {width}×{height} 像素。\n'
                  f'bbox 使用四个顺时针顶点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，'
                  f'每点满足 0≤x≤{width}、0≤y≤{height}；支持倾斜矩形或凸四边形，不能自交或面积为零。\n'
                  '沿边界依次给点，不按对角线跳跃。水平矩形正确顺序是左上→右上→右下→左下，'
                  '例如 [[100,100],[180,100],[180,140],[100,140]]；'
                  '倾斜矩形的格式例子是 [[100,100],[180,140],[160,180],[80,140]]。这些只是格式例子，不是当前图像候选。\n'
                  '半透明绿色 mask 和黄色边界是预测提示，不是真实地物；不选择主定位模式。\n'
                  'area_id 从 Area_1 开始按搜索顺序递增；只输出 API schema 定义的五个区域字段。'
                  'matched_rules 至少包含一项实际支撑选区的规则，rule_id 和 rule_name 必须照抄 Rule 的 id/title，'
                  '不得编造 ID、改写规则名或为凑数量引用无依据规则；多个命中可以全部列出且不重复。\n'
                  '所有 bbox 的内部必须互不重叠，边界接触允许。优先用贴合道路走向的倾斜矩形；必要时缩小、拆分或合并候选，'
                  '不为不同定位视角重复选择同一片地表。不用覆盖大范围背景的水平框代替斜向路段。\n'
                  '道路 Mask 和 polygon 仅作参考，不是选区门槛；原图中你认为是道路或可能可通行的区域都可作为搜索主线，'
                  '包括土路、田间便道、场地通道、开阔硬化场地和连续裸露地表，不要求正式道路形态或 Mask 命中。'
                  '每个 ROI 应包含上述道路或潜在通行区域及相关环境；潜在通行需有可见地表、连续性或障碍线索支撑，'
                  '在既有描述或理由字段中保留不确定性，不宣称已验证车辆实际可通行。\n'
                  + ('候选数量不设配额，不凑数；无可靠候选时输出 {"areas": []}。' if limit is None
                     else f'调用方设置候选数量上限 {limit}，这是最多而不是必须输出的数量；不得为凑数量编造区域。'))
        attempts = []
        overlap_resolution = None
        write_json(directory / 'inference.json', {'attempts': attempts, 'status': 'started'})
        # 仅业务/结构校验失败时最多要求模型纠正一次；不重试传输或服务错误。
        for attempt in range(1, 3):
            attempt_dir = directory / 'attempts' / f'{attempt:02d}'
            attempt_dir.mkdir(parents=True)
            request_path = attempt_dir / 'request.json'
            response_path = attempt_dir / 'response.json'
            response = client._infer(prompt, [encoded], system=system, max_tokens=max_tokens,
                                     temperature=0.0, json_output=True, output_schema=schema,
                                     request_path=request_path, response_path=response_path)
            (attempt_dir / 'content.txt').write_text(response.text, encoding='utf-8')
            details = {'attempt': attempt, 'request_id': response.request_id, 'usage': response.usage,
                       'latency_seconds': response.latency_seconds, 'request_path': str(request_path),
                       'response_path': str(response_path), 'content_path': str(attempt_dir / 'content.txt')}
            data = None
            try:
                data = _parse(response.text)
                validate_candidates(data, config['schema'], width, height, limit)
            except PositioningError as exc:
                details['validation_error'] = str(exc)
                attempts.append(details)
                write_json(directory / 'inference.json', {'attempts': attempts, 'status': 'validation_failed'})
                if attempt == 2:
                    if data is None:
                        raise
                    # 只处理剩余面积重叠；字段、规则引用、顶点形状等错误仍明确失败。
                    validate_candidates(data, config['schema'], width, height, limit, check_overlap=False)
                    data, overlap_resolution = select_disjoint_candidates(data)
                    validate_candidates(data, config['schema'], width, height, limit)
                    details['resolution'] = 'nonoverlapping_subset_selected'
                    break
                prompt += ('\n上一次输出未通过程序校验：' + str(exc) +
                           '\n上一次 JSON：' + response.text +
                           '\n请重新检查图像并输出一份完整替代结果；消除面积重叠、无效四边形、越界或编号问题，'
                           '不要复制上一次的错误 bbox。先修正每个无效四边形的顶点顺序，再逐对消除重叠；'
                           '无法可靠划分的候选应舍弃，允许减少数量，不保留无视觉依据的区域，不为达到数量上限复制候选。')
                continue
            attempts.append(details)
            write_json(directory / 'inference.json', {'attempts': attempts, 'status': 'validated'})
            break
        if (_digest(state['image_path']) != state['image_digest'] or
                _digest(str(enhanced_path)) != enhanced_digest):
            raise PositioningError('候选推理期间输入影像发生改变')
        mapped = copy.deepcopy(data)
        for area in mapped['areas']:
            area['bbox'] = [[x * original_width / width, y * original_height / height]
                            for x, y in region_vertices(area['bbox'])]
        # 保留浮点顶点，不向外取整，避免本来相邻的区域在映射后产生重叠。
        validate_candidates(mapped, config['schema'], original_width, original_height, limit)
        output = resolve_path(state['trace_dir'])
        candidate_path, original_path = output / 'candidates.json', output / 'candidates_original.json'
        write_json(candidate_path, data)
        write_json(original_path, mapped)
        inference = {'usage': response.usage, 'latency_seconds': sum(item['latency_seconds'] for item in attempts),
                     'attempts': attempts,
                     'overlap_resolution': overlap_resolution, 'nonoverlap_validated': True,
                     'request_id': response.request_id, 'finish_reason': response.finish_reason,
                     'api': client.config.base_url + '/chat/completions', 'response_format': 'json_schema',
                     'schema_strict': True, 'backend': 'xgrammar', 'image_count': 1,
                     'server_log': str(client.runtime.log_path) if client.runtime.log_path else None}
        write_json(directory / 'inference.json', inference)
        context['coordinate_mapping'] = {'areas': [
            {'area_id': source['area_id'], 'vlm_bbox': source['bbox'], 'original_bbox': target['bbox']}
            for source, target in zip(data['areas'], mapped['areas'])]}
        write_json(directory / 'context.json', context)
        traces = record_trace(state, 'identify_candidates', {
            'status': 'completed', 'candidate_count': len(data['areas']), 'input_view': view,
            'candidate_json_path': str(candidate_path), 'candidate_original_json_path': str(original_path),
            'runtime_dir': str(directory.parent.resolve()), 'prompt_loading': metadata, 'inference': inference})
        return relative_paths({'model_candidates': data, 'result': mapped, 'candidate_view': view, 'inference': inference,
                'candidate_json_path': str(candidate_path), 'candidate_original_json_path': str(original_path),
                'runtime_dir': str(directory.parent.resolve()), **traces})
    except Exception as exc:
        failure = {'status': 'failed', 'error': str(exc), 'latency_seconds': time.monotonic() - started,
                   'runtime_dir': str(directory.resolve())}
        write_json(directory / 'error.json', failure)
        record_trace(state, 'identify_candidates', failure)
        raise
