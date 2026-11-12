"""整图拓扑布块流程配置（Node 1-5）；通用字段继承自 PreprocessConfig。"""
from __future__ import annotations

from dataclasses import dataclass, field
from .config import PreprocessConfig


NODE_ORDER = ('load_image_and_resolution', 'classify_scene_type', 'extract_road_graph', 'clean_road_graph', 'build_blocks_and_groups',
              'compute_block_features', 'vlm_review_blocks', 'vlm_expand_blocks', 'fuse_degraded_patches', 'recommend_localization')
# 流程算法版本：会改变节点结果的修改都要递增。写入 state.json 与 manifest，不同版本的结果不能直接比较。
# 2：Node 5 高度 0 点只取语义路面、语义改为裁剪整图结果；Node 3 面状区域改用距离变换（与原膨胀差异约 0.001% 像元）
# 3：DSM 高度分段改为 0-0.5 / 0.5-2 / >2 m（原 0.5-3 / >3 m），Node 6/7 高度图配色随之变化
# 4：Node 5 去掉可靠性标记（reliability / *_unreliable_blocks / *_dominant）及其配置，final_summary 不再把这些块算作不完整
# 5：Node 7 基准范围不完整的已确认退化块不再跳过，保留 Node 6 实际判断范围进入 Node 8 融合；Node 6 不退化的块不再记为未融合问题
# 6：基准块 64 → 75 m、Node 7 扩展档改为 100 m；Node 6 语义复核输入改为局部 RGB（无标记）+ 局部语义 + 2 倍范围语义上下文三图；语义图道路改为红色（原白色），Node 7 语义中心线改为黄色；
#    Node 6 高度复核输入改为 100 m 整体 RGB（目标框）+ 局部 RGB（无标记）+ 局部 DSM 伪彩色（DSM − 100 m 范围 P20，固定 [-2, 8] m 线性 Viridis）；
#    Node 7 高度图改用同一着色（去掉 Node 6/7 可比性检查），Node 7 输入图不再画道路中心线；final_summary 不再把 Node 7 请求失败算作不完整（基准已保留进入融合）
# 7：Node 6 语义上下文改为 100 m，至少覆盖整个局部 Patch。
# 8：Node 5 路面基准改用中心 5 m 或 10 m 方窗内的路面 DSM 中位数。
# 9：Node 6 提示词明确语义及高程 BEV 的匹配对象和退化机制。
# 10：精简 Node 6 提示词，并明确两条支路的遮挡证据如何进入判定。
# 11：Node 6 按实际输入图调整观察步骤，并说明专题图处理和程序标记。
# 12：Node 6/7 高程输出增加三个按问题组织的证据字段。
# 13：Node 6/7 高程结论采用模型 verdict。
# 14：Node 6/7 要求模型用当前窗口已记录的图像证据支持 verdict。
# 15：Node 6/7 校验结论证据引用和数据可用性，扩展独立观察。
# 16：Node 7 明确退化范围扩展任务，并按实际双图与档位提供输入说明。
# 17：Node 7 补充语义和高程扩展窗判据，明确新增环带与整窗证据的关系。
# 18：Node 7 默认只评估 100 m 扩展档，与正式流程配置一致。
# 19：Node 9 按 P0/P1 比较定位方式，并明确短时轮速过渡和 GPS 开阔条件。
# 20：Node 4 沿拓扑按固定 5% 名义弧长重叠布块，去重保留精确道路覆盖。
# 21：独立语义/高程送审、缺失数据汇总、隔离单节点诊断，并校验定位推荐条件。
# 22：Node 6 只复核 Node 5 规则命中的支路，缺路面基准单独报告。
# 23：Node 5 拼图增加简洁的比例及初筛结果栏。
# 24：正式配置启用 Node 6 的 medium 思考档。
# 25：本地 VLM 默认五路并行，并行时默认关闭草稿模型。
# 26：正式流程的所有 VLM 节点启用 medium 思考。
PIPELINE_VERSION = 26
# Node 8 可融合的退化类别；non_degraded、unknown 不属于已确认的退化，不能配置进融合
FUSABLE_DEGRADATION_TYPES = ('degradation', 'longitudinal_degradation')


@dataclass(frozen=True)
class GlobalPreprocessConfig(PreprocessConfig):
    workflow: str = 'global_blocks'
    segmenter: str = 'magroad_scene'
    tile_grid: int = 3
    cache_enabled: bool = True
    cache_dir: str = 'output/map_preprocess/cache'
    dummy_graph_path: str | None = None
    # segmenter=magroad_scene：Node 1b 由本地 VLM 看整图判断道路结构（结构化=urban / 非结构化=wild），Node 2 按场景选 MaGRoad 权重。
    # 写法同 agent.run_magroad --checkpoints：权重名，可附阈值覆盖。
    magroad_wild_checkpoint: str = 'wildroad'
    magroad_urban_checkpoint: str = 'globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455'
    scene_type_override: str | None = None        # wild/urban：跳过 VLM 直接指定（离线调试或人工确认）
    # 已确认的图：图片文件名（不含扩展名）→ wild/urban，不调用 VLM；优先级 override > known > VLM
    scene_type_known: dict[str, str] = field(default_factory=dict)
    scene_type_fallback: str = 'wild'             # VLM 失败、判“不确定”或置信度低时使用
    scene_vlm_image_px: int = 1024                # 整图缩略图长边
    scene_vlm_thinking_budget: int = 0
    scene_vlm_reasoning_effort: str | None = None
    scene_vlm_max_tokens: int = 2560
    scene_vlm_temperature: float = 0.0
    # 整图语义推理（语义路线和 Node 6/7 共用，按输入哈希缓存）：工作分辨率，原图更粗时不上采样；分窗大小（模型单次输入上限 1600 万像元）
    semantic_roads_resolution_m: float = 0.05
    semantic_roads_window_px: int = 3072
    semantic_roads_window_margin_px: int = 128
    # 掩码拓扑：由 MaGRoad 概率和语义分割生成路网（utils/road_topo.py，参数在 road_topo_config），代替旧的逐步清洗；
    # 开启后下面的 min_component_length_m ~ road_smooth_sigma_m、min_edge_score 不再生效。默认关闭，两份流程 YAML 开启。
    road_topo_enabled: bool = False
    road_topo_config: str = 'agent/map_preprocess/pipeline.yaml'
    # 流程跑到哪个节点为止（NODE_ORDER 中的名字）；None 表示跑完整流程
    run_until: str | None = None
    min_component_length_m: float = 20.0
    # 过近路口合并：两端都是路口（度 >= 3）且短于该值（米）的边，两端合并为一个路口；null = block_size_m/4
    #（64 m 块为 16 m，约一个路口的宽度，再大会把两个真实路口并成一个）；0 关闭。
    junction_merge_dist_m: float | None = None
    min_spur_length_m: float = 5.0
    # 并行去重：一条路至少 parallel_min_overlap 比例的长度与另一条路相距 <= parallel_merge_dist_m（米）且局部方向
    # 夹角 <= parallel_max_angle_deg 时去掉一条；河西宽路两幅车道被提成两条线时相距 11~18 m，超过 20 m 开始误删
    # 真实相邻的路（KITTI/汨罗）。距离 0 关闭。
    parallel_merge_dist_m: float = 18.0
    parallel_max_angle_deg: float = 20.0
    parallel_min_overlap: float = .5
    # 近距连通：悬空路端离别的路 <= 该值（米）就接上；0 关闭。
    road_connect_dist_m: float = 3.0
    # 短路删除：一端悬空且短于 short_road_max_m（米）的边删除；离开路口的最远距离不到 small_loop_max_reach_m（米，
    # 默认半个 64 m 块：整圈都在路口那一块里）的自环删除；0 关闭对应项。
    short_road_max_m: float = 10.0
    small_loop_max_reach_m: float = 32.0
    # 贯通路网：每条路都要有进有出。逐层删断头路（端点度 1 且离图边超过 road_border_margin_m，贴图边的路端算通向图外），
    # 只留总长最大的连通分量，合并相隔不足 road_close_node_dist_m 的路口，打开周长不足 small_cycle_max_length_m 的小环。
    # 默认关闭（合成/离线道路通常有断头路），两份流程 YAML 中开启；后三个距离为 0 时关闭对应项。
    road_through_only: bool = False
    road_border_margin_m: float = 10.0
    road_close_node_dist_m: float = 16.0
    small_cycle_max_length_m: float = 120.0
    # 贯通路网内的重复路去弱：两条边首尾相同时，沿线模型道路概率 >= road_duplicate_prob 的长度比例低于
    # road_duplicate_min_support 的边删除，最强的一条始终保留；分割器没有概率图时跳过；后者为 0 关闭。
    road_duplicate_prob: float = .18
    road_duplicate_min_support: float = .6
    # 支路贴着干路走到路口的长度不小于该值（米）时，在真实分岔处建节点，重合段只留一条；0 关闭。
    road_shared_min_length_m: float = 0.
    # 限制节点接的路数：超过 road_max_node_degree 的路口拆成相距 road_max_shared_gap_m 的两个节点，短于该间距的边先合并；间距为 0 关闭。
    road_max_shared_gap_m: float = 0.
    road_max_node_degree: int = 4
    # MaGRoad 为主路网，用概率骨架补漏：长度不小于该值（米）、概率强、不压建筑、两端能接入路网的路段补回来；0 关闭。
    road_skeleton_fill_min_length_m: float = 0.
    # 内部路面占比不低于前者、周长不超过后者（米）的环是宽路口网格或分隔带双线：紧凑的收成一个路口，细长的去掉一条边；
    # 需要语义路线的整图语义，前者为 0 关闭。真实街区中间是房屋绿地（KITTI 最高 0.42），河西路口网格 0.8~1。
    road_paved_ring_min_road: float = 0.
    road_paved_ring_max_perimeter_m: float = 300.
    # 语义路线：拓扑定稿后路线改走语义道路（流程语义模型的 road 类）中线，连接关系不变，随后再做曲率平滑。
    # corridor 是路线离原线的上限（米）；node_snap 是路口吸附到路面中心的半径（米）。默认关闭，两份流程 YAML 中开启。
    road_semantic_route_enabled: bool = False
    road_semantic_route_corridor_m: float = 15.0
    road_semantic_route_node_snap_m: float = 8.0
    # 曲率平滑：沿弧长高斯平滑的 sigma（米），90° 转角磨成半径约 1.6×sigma 的圆弧；节点不动；0 关闭。
    road_smooth_sigma_m: float = 6.0
    score_length_scale_m: float = 100.0
    edge_score_weights: dict[str, float] = field(default_factory=lambda: {
        'length': .25, 'confidence': .4, 'straightness': .2, 'connectivity': .15})
    min_edge_score: float = .2
    keep_main_trunk_only: bool = False
    min_attached_edge_length_m: float = 20.0
    route_polyline: tuple[tuple[float, float], ...] = ()
    route_coordinate_system: str = 'pixel'
    roi_mask: str | None = None
    route_buffer_m: float = 50.0
    constraint_sample_spacing_m: float = 1.0
    branch_review_enabled: bool = False
    review_top_n_edges: int = 10
    reviewer_type: str = 'pass_through'
    block_size_m: float = 75.0
    dedup_enabled: bool = True
    # walk：直接去除初始布块中的高重叠冗余框；greedy：独立候选全局贪心覆盖。
    dedup_method: str = 'walk'
    # 兼容旧配置：当前 walk 按精确道路区间保护覆盖，不使用最小新增长度。
    dedup_min_gain_m: float = 20.0
    dedup_candidate_spacing_m: float = 10.0
    # walk 使用精确区间剪除；greedy 使用以下独占长度阈值和 min_road_coverage。
    dedup_prune_enabled: bool = True
    dedup_prune_max_loss_m: float = 5.0
    road_sample_spacing_m: float = 1.0
    min_road_coverage: float = .995
    block_overlap_ratio: float = .05
    blocks_per_group: int = 3
    group_overlap_blocks: int = 1
    long_edge_strategy: str = 'split_continuous'
    context_size_m: float | None = 300.0
    max_blocks_total: int | None = None
    debug_block_ids: tuple[str, ...] = ()
    # Node 5：每块（图内部分）的语义（RGB）与高程（DSM，可选）。没有 DSM 时只判语义退化。
    block_features_enabled: bool = False
    perception_config: str = 'agent/perception_tools.yaml'
    # 仅为兼容旧配置保留：Node 5 已改为裁剪整图语义，不再逐块缓存
    feature_cache_enabled: bool = True
    feature_cache_dir: str = 'output/map_preprocess/cache/block_features'
    feature_min_class_confidence: float = .5
    # 兼容旧配置；Node 5 路面基准已改用 5 m / 10 m 方窗。
    feature_center_radius_m: float = 2.0
    # 低于 0 点超过该值（米）的像元不算有效像元；[-该值, 0) 视为 0 m（路面 DSM 噪声）。设 0 为严格以 0 为界。
    feature_below_base_tolerance_m: float = 0.3
    # 高度连续性：水平 feature_step_run_m（米）内高差 >= feature_step_height_m（米）视为台阶（阶跃）。
    # 从块中心路面出发不越过台阶能到达的低处 = 缓坡低于（正常地面，计入最低段）；隔着台阶的低处 = 阶跃低于（不合理的低）。
    feature_step_height_m: float = 0.5
    feature_step_run_m: float = 1.0
    # 相对路面高度（米）的分段边界，(0.5, 2) → 0-0.5 m 路面/地面 / 0.5-2 m 低矮物 / >2 m 建筑、树或围墙等高出结构；高于 0.5 m 即可用于高程匹配。
    feature_dsm_height_bands_m: tuple[float, ...] = (0.5, 2.0)
    # 可选去坡（默认关）：分 N×N 格取低分位拟合地形平面，先减去再取中心路面为 0 点。
    feature_dsm_detrend: bool = False
    feature_dsm_ground_cells: int = 8
    # 每块高度图的色带上限（米，相对路面高度）。
    feature_height_vis_max_m: float = 15.0
    # 高度退化（需要 DSM），任一满足即退化：最低段（默认 0-0.5m）> feature_height_flat_min_pct（几乎全和路面一样平，缓坡低于计入此段），
    # 或阶跃低于路面的像元 step_below_pct > feature_height_below_max_pct（如路堤/高架，四周隔着台阶比路低）。>2m 占比大不算退化。
    feature_height_flat_min_pct: float = 90.
    feature_height_below_max_pct: float = 70.
    # 语义退化：道路占比 < feature_semantic_max_road_pct 且 植被占比 > feature_semantic_min_plant_pct（%）。
    feature_semantic_max_road_pct: float = 10.
    feature_semantic_min_plant_pct: float = 65.
    # 每块输出 rgb/semantic/height/height_bands PNG 到 visualization/07_per_block_images/；null 为原分辨率。
    feature_vis_enabled: bool = True
    feature_vis_max_side: int | None = 1000
    # Node 6：退化块 VLM 复核（本地 Qwen）。语义退化块走 semantic 复核、高度退化块走 height 复核（两套独立 prompt 系统，
    # 见 agent/vlm/block_review/）。VLM 不知道 Node 5 的结论，独立判断；判定不可行驶的块直接删除。
    vlm_review_enabled: bool = False
    # 每张输入图的边长（像素），图像覆盖当前块（block_size_m 见方）
    vlm_review_image_px: int = 768
    # 兼容历史配置；Node 6 v2 不再向 VLM 发送未在图上标出的高度网格。
    vlm_review_grid_cells: int = 6
    # 兼容历史配置；Node 6 v2 不再使用建筑占比自动豁免。
    vlm_review_semantic_min_building_pct: float = 10.
    # height 复核：整体 RGB 上下文边长（米，>= block_size_m）；参考高程 H_ref 取该范围内有效 DSM 的百分位（只是着色基准，不是路面高程）；
    # 局部 DSM 图按 DSM − H_ref 的固定物理范围（米）用 Viridis 连续着色，所有块同一色标，超出范围按端点显示并记录饱和比例。
    vlm_review_height_context_m: float = 100.
    vlm_review_dsm_ref_percentile: float = 20.
    vlm_review_dsm_range_m: tuple[float, float] = (-2., 8.)
    # Qwen thinking：0 = 关闭（默认）；正整数 = 到额度强制结束思考；None = 不限，由模型自然结束。
    vlm_review_thinking_budget: int | None = 0
    vlm_review_reasoning_effort: str | None = None
    vlm_review_max_tokens: int = 8192
    vlm_review_workers: int = 5
    # 只渲染输入图与 prompt，不调用 VLM、不修改 manifest
    vlm_review_dry_run: bool = False
    # Node 7：退化区域尺寸扩展（本地 Qwen）。基准为 block_size_m，同心逐档扩大，semantic/height 独立记录；需要 Node 6。
    vlm_expansion_enabled: bool = False
    vlm_expansion_sizes_m: tuple[float, ...] = (100.,)
    # 发给 VLM 的图像边长（像素）；不超过客户端 VLM_MAX_IMAGE_PIXELS 时不再二次缩放
    vlm_expansion_image_px: int = 1024
    # thinking 语义同 vlm_review_thinking_budget，默认关闭
    vlm_expansion_thinking_budget: int | None = 0
    vlm_expansion_reasoning_effort: str | None = None
    vlm_expansion_max_tokens: int = 12288
    vlm_expansion_workers: int = 5
    vlm_expansion_dry_run: bool = False
    # Node 8：退化 Patch 按正面积重叠融合（纯几何，不调用 VLM）；需要 Node 7。
    degradation_fusion_enabled: bool = False
    degradation_fusion_types: tuple[str, ...] = FUSABLE_DEGRADATION_TYPES
    # True 时不同退化类型之间也建立融合关系
    degradation_fusion_cross_type: bool = False
    # Node 9：逐融合区域推荐辅助定位方式（本地 Qwen）；需要 Node 8。
    vlm_localization_enabled: bool = False
    # 裁剪图长边（像素），保持长宽比；不超过客户端 VLM_MAX_IMAGE_PIXELS 时不再二次缩放
    vlm_localization_image_px: int = 1024
    # thinking 语义同 vlm_review_thinking_budget，默认关闭
    vlm_localization_thinking_budget: int | None = 0
    vlm_localization_reasoning_effort: str | None = None
    vlm_localization_max_tokens: int = 4096
    vlm_localization_workers: int = 2
    # 输出无效或服务瞬时错误时的额外尝试次数；0 = 不重试
    vlm_localization_max_retries: int = 1
    # 只渲染输入图与 prompt，不调用 VLM、不修改 manifest
    vlm_localization_dry_run: bool = False

    @classmethod
    def from_recorded(cls, recorded: dict, *, keep_features: bool = True) -> 'GlobalPreprocessConfig':
        """从 manifest 记录的配置重建：忽略已删除的字段；keep_features=False 时 Node 5 参数取当前默认值。"""
        import dataclasses
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in recorded.items()
                      if k in names and (keep_features or not k.startswith('feature_'))})

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ('scene_vlm_reasoning_effort', 'vlm_review_reasoning_effort',
                     'vlm_expansion_reasoning_effort', 'vlm_localization_reasoning_effort'):
            if getattr(self, name) not in (None, 'low', 'medium', 'xhigh'):
                raise ValueError(f'{name} 必须为 low、medium、xhigh 或 null')
        self._validate_features()
        if self.workflow != 'global_blocks' or self.segmenter not in {'magroad_scene', 'dummy'}:
            raise ValueError('整图模式 workflow=global_blocks，segmenter=magroad_scene/dummy')
        self._validate_scene()
        for name in ('tile_grid', 'review_top_n_edges', 'group_overlap_blocks'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} 必须为非负整数')
        for name in ('blocks_per_group',):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} 必须为正整数')
        if self.group_overlap_blocks >= self.blocks_per_group:
            raise ValueError('group_overlap_blocks 必须小于 blocks_per_group')
        for name in ('block_size_m', 'score_length_scale_m', 'constraint_sample_spacing_m',
                     'dedup_candidate_spacing_m', 'road_sample_spacing_m'):
            self._number(name, getattr(self, name), positive=True)
        for name in ('min_component_length_m', 'min_spur_length_m', 'parallel_merge_dist_m',
                     'min_attached_edge_length_m', 'route_buffer_m', 'parallel_max_angle_deg',
                     'road_connect_dist_m', 'short_road_max_m', 'small_loop_max_reach_m', 'road_smooth_sigma_m',
                     'road_border_margin_m', 'road_close_node_dist_m', 'small_cycle_max_length_m',
                     'road_semantic_route_corridor_m', 'road_semantic_route_node_snap_m', 'road_shared_min_length_m',
                     'road_paved_ring_max_perimeter_m', 'road_max_shared_gap_m', 'road_skeleton_fill_min_length_m'):
            self._number(name, getattr(self, name))
            if getattr(self, name) < 0:
                raise ValueError(f'{name} 不能为负')
        for name in ('road_paved_ring_min_road', 'block_overlap_ratio', 'min_edge_score', 'parallel_min_overlap', 'road_duplicate_prob',
                     'road_duplicate_min_support'):
            self._number(name, getattr(self, name))
            if not 0 <= getattr(self, name) <= 1 or (name == 'block_overlap_ratio' and getattr(self, name) == 1):
                raise ValueError(f'{name} 超出范围')
        if self.parallel_max_angle_deg > 90:
            raise ValueError('parallel_max_angle_deg 必须 <= 90')
        if self.parallel_min_overlap == 0:
            raise ValueError('parallel_min_overlap 必须在 (0, 1]')
        self._number('min_road_coverage', self.min_road_coverage, positive=True)
        if self.min_road_coverage > 1:
            raise ValueError('min_road_coverage 必须在 (0, 1]')
        if self.junction_merge_dist_m is not None:
            self._number('junction_merge_dist_m', self.junction_merge_dist_m)
            if self.junction_merge_dist_m < 0:
                raise ValueError('junction_merge_dist_m 不能为负或应为 null')
        if self.context_size_m is not None:
            self._number('context_size_m', self.context_size_m, positive=True)
        if self.max_blocks_total is not None and (type(self.max_blocks_total) is not int or self.max_blocks_total < 0):
            raise ValueError('max_blocks_total 必须为非负整数或 null')
        if self.dedup_method not in {'walk', 'greedy'}:
            raise ValueError('dedup_method 必须为 walk 或 greedy')
        self._number('dedup_prune_max_loss_m', self.dedup_prune_max_loss_m)
        if self.dedup_prune_max_loss_m < 0:
            raise ValueError('dedup_prune_max_loss_m 不能为负')
        self._number('dedup_min_gain_m', self.dedup_min_gain_m)
        if self.dedup_min_gain_m < 0:
            raise ValueError('dedup_min_gain_m 不能为负')
        if self.long_edge_strategy not in {'split_continuous', 'sparse3'}:
            raise ValueError('非法 long_edge_strategy')
        if self.reviewer_type not in {'pass_through', 'rule_based'}:
            raise ValueError('非法 reviewer_type')
        if self.route_coordinate_system not in {'pixel', 'lonlat'}:
            raise ValueError('route_coordinate_system 必须为 pixel 或 lonlat')
        if self.route_polyline:
            if len(self.route_polyline) < 2:
                raise ValueError('route_polyline 至少两个点')
            for point in self.route_polyline:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError('route_polyline 必须为坐标对')
                for value in point:
                    self._number('route_polyline', value)
                if self.route_coordinate_system == 'lonlat' and (abs(point[0]) > 180 or abs(point[1]) > 90):
                    raise ValueError('route_polyline 经纬度越界')
        keys = {'length', 'confidence', 'straightness', 'connectivity'}
        if not isinstance(self.edge_score_weights, dict) or set(self.edge_score_weights) != keys:
            raise ValueError(f'edge_score_weights 需要 {sorted(keys)}')
        for value in self.edge_score_weights.values():
            self._number('edge_score_weights', value)
            if value < 0:
                raise ValueError('评分权重不能为负')
        if sum(self.edge_score_weights.values()) <= 0:
            raise ValueError('评分权重和必须为正')
        self._number('semantic_roads_resolution_m', self.semantic_roads_resolution_m, positive=True)
        if type(self.semantic_roads_window_px) is not int or not 512 <= self.semantic_roads_window_px <= 3800:
            raise ValueError('semantic_roads_window_px 必须为 512~3800 的整数')
        if type(self.semantic_roads_window_margin_px) is not int or not 0 <= self.semantic_roads_window_margin_px <= 256:
            raise ValueError('semantic_roads_window_margin_px 必须为 0~256 的整数')
        for name in ('cache_enabled', 'branch_review_enabled', 'keep_main_trunk_only', 'dedup_enabled',
                     'dedup_prune_enabled', 'road_through_only', 'road_topo_enabled',
                     'road_semantic_route_enabled'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} 必须为布尔值')
        for name in ('dummy_graph_path', 'roi_mask'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{name} 必须为路径或 null')
        if self.run_until is not None and self.run_until not in NODE_ORDER:
            raise ValueError(f'run_until 必须是 {NODE_ORDER} 之一')
        if not isinstance(self.road_topo_config, str) or not self.road_topo_config.strip():
            raise ValueError('road_topo_config 必须为路径')
        if not isinstance(self.cache_dir, str) or not self.cache_dir.strip():
            raise ValueError('cache_dir 必须为路径')
        if not isinstance(self.debug_block_ids, (list, tuple)) or any(not isinstance(x, str) or not x for x in self.debug_block_ids):
            raise ValueError('debug_block_ids 必须为 ID 列表')

    def _validate_scene(self) -> None:
        from percection_tool.magroad.inference import parse_checkpoint_spec
        for name in ('magroad_wild_checkpoint', 'magroad_urban_checkpoint'):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'{name} 必须为字符串，如 wildroad 或 globalscale:ROAD=0.15')
            parse_checkpoint_spec(getattr(self, name))
        if self.scene_type_override not in (None, 'wild', 'urban'):
            raise ValueError('scene_type_override 必须为 wild、urban 或 null')
        if not isinstance(self.scene_type_known, dict) or any(
                not isinstance(k, str) or not k or v not in ('wild', 'urban') for k, v in self.scene_type_known.items()):
            raise ValueError('scene_type_known 必须为 图片名 → wild/urban 的映射')
        if self.scene_type_fallback not in ('wild', 'urban'):
            raise ValueError('scene_type_fallback 必须为 wild 或 urban')
        for name, low in (('scene_vlm_image_px', 256), ('scene_vlm_max_tokens', 256)):
            if type(getattr(self, name)) is not int or getattr(self, name) < low:
                raise ValueError(f'{name} 必须为 >= {low} 的整数')
        self._number('scene_vlm_temperature', self.scene_vlm_temperature)
        if not 0 <= self.scene_vlm_temperature <= 2:
            raise ValueError('scene_vlm_temperature 必须在 [0, 2]')
        budget = self.scene_vlm_thinking_budget
        if type(budget) is not int or budget < 0:
            raise ValueError('scene_vlm_thinking_budget 必须为 >= 0 的整数（0 = 关闭 thinking）')
        if budget and budget + 512 > self.scene_vlm_max_tokens:
            raise ValueError('scene_vlm_max_tokens 至少要比 scene_vlm_thinking_budget 多 512')

    def _validate_localization(self) -> None:
        for name in ('vlm_localization_enabled', 'vlm_localization_dry_run'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} 必须为布尔值')
        if self.vlm_localization_enabled and not self.degradation_fusion_enabled:
            raise ValueError('vlm_localization_enabled 需要 degradation_fusion_enabled（推荐对象是 Node 8 的融合区域）')
        for name, low in (('vlm_localization_image_px', 128), ('vlm_localization_max_tokens', 256), ('vlm_localization_workers', 1),
                          ('vlm_localization_max_retries', 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < low:
                raise ValueError(f'{name} 必须为 >= {low} 的整数')
        budget = self.vlm_localization_thinking_budget
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError('vlm_localization_thinking_budget 必须为 None（不限）或 >= 0 的整数（0 = 关闭 thinking）')
        if budget and budget + 512 > self.vlm_localization_max_tokens:
            raise ValueError('vlm_localization_max_tokens 至少要比 vlm_localization_thinking_budget 多 512')

    def _validate_expansion(self) -> None:
        if self.vlm_expansion_enabled and not self.vlm_review_enabled:
            raise ValueError('vlm_expansion_enabled 需要 vlm_review_enabled（扩展对象是 Node 6 确认的退化区域）')
        for name, low in (('vlm_expansion_image_px', 128), ('vlm_expansion_max_tokens', 256), ('vlm_expansion_workers', 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < low:
                raise ValueError(f'{name} 必须为 >= {low} 的整数')
        budget = self.vlm_expansion_thinking_budget
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError('vlm_expansion_thinking_budget 必须为 None（不限）或 >= 0 的整数（0 = 关闭 thinking）')
        if budget and budget + 512 > self.vlm_expansion_max_tokens:
            raise ValueError('vlm_expansion_max_tokens 至少要比 vlm_expansion_thinking_budget 多 512')
        sizes = self.vlm_expansion_sizes_m
        if not isinstance(sizes, (list, tuple)) or not sizes:
            raise ValueError('vlm_expansion_sizes_m 至少一个扩展尺寸（米）')
        for value in sizes:
            self._number('vlm_expansion_sizes_m', value, positive=True)
        if any(a >= b for a, b in zip(sizes, sizes[1:])):
            raise ValueError('vlm_expansion_sizes_m 必须严格递增')
        if self.vlm_expansion_enabled and sizes[0] <= self.block_size_m:
            raise ValueError('vlm_expansion_sizes_m 必须都大于基准尺寸 block_size_m')

    def _validate_features(self) -> None:
        for name in ('vlm_review_enabled', 'vlm_review_dry_run', 'vlm_expansion_enabled', 'vlm_expansion_dry_run',
                     'degradation_fusion_enabled', 'degradation_fusion_cross_type'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} 必须为布尔值')
        if self.degradation_fusion_enabled and not self.vlm_expansion_enabled:
            raise ValueError('degradation_fusion_enabled 需要 vlm_expansion_enabled（融合对象是 Node 7 的退化 Patch）')
        types = self.degradation_fusion_types
        if not isinstance(types, (list, tuple)) or not types or any(x not in FUSABLE_DEGRADATION_TYPES for x in types):
            raise ValueError(f'degradation_fusion_types 必须为 {FUSABLE_DEGRADATION_TYPES} 中的非空子集，当前为 {types!r}')
        self._validate_localization()
        for name, low in (('vlm_review_image_px', 128), ('vlm_review_grid_cells', 2), ('vlm_review_max_tokens', 256),
                          ('vlm_review_workers', 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < low:
                raise ValueError(f'{name} 必须为 >= {low} 的整数')
        if not 0 < self.vlm_review_semantic_min_building_pct <= 100:
            raise ValueError('vlm_review_semantic_min_building_pct 必须在 (0, 100] 内')
        self._number('vlm_review_height_context_m', self.vlm_review_height_context_m, positive=True)
        if self.vlm_review_height_context_m < self.block_size_m:
            raise ValueError('vlm_review_height_context_m 不能小于 block_size_m（上下文要包含目标 Patch）')
        self._number('vlm_review_dsm_ref_percentile', self.vlm_review_dsm_ref_percentile)
        if not 0 <= self.vlm_review_dsm_ref_percentile <= 100:
            raise ValueError('vlm_review_dsm_ref_percentile 必须在 [0, 100] 内')
        span = self.vlm_review_dsm_range_m
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError('vlm_review_dsm_range_m 必须为 [下限, 上限]（米）')
        for value in span:
            self._number('vlm_review_dsm_range_m', value)
        if not span[0] < span[1]:
            raise ValueError('vlm_review_dsm_range_m 下限必须小于上限')
        budget = self.vlm_review_thinking_budget
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError('vlm_review_thinking_budget 必须为 None（不限）或 >= 0 的整数（0 = 关闭 thinking）')
        if budget and budget + 512 > self.vlm_review_max_tokens:
            raise ValueError('vlm_review_max_tokens 至少要比 vlm_review_thinking_budget 多 512，给最终 JSON 留出空间')
        if self.vlm_review_enabled and not self.block_features_enabled:
            raise ValueError('vlm_review_enabled 需要 block_features_enabled（复核对象是 Node 5 的退化块）')
        self._validate_expansion()
        if self.feature_vis_max_side is not None and (type(self.feature_vis_max_side) is not int or self.feature_vis_max_side < 16):
            raise ValueError('feature_vis_max_side 必须为 >=16 的整数或 null')
        for name in ('block_features_enabled', 'feature_cache_enabled', 'feature_vis_enabled', 'feature_dsm_detrend'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} 必须为布尔值')
        for name in ('perception_config', 'feature_cache_dir'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} 必须为路径')
        for name in ('feature_height_flat_min_pct', 'feature_height_below_max_pct', 'feature_semantic_max_road_pct',
                     'feature_semantic_min_plant_pct'):
            self._number(name, getattr(self, name))
            if not 0 < getattr(self, name) < 100:
                raise ValueError(f'{name} 必须在 (0, 100)')
        if type(self.feature_dsm_ground_cells) is not int or not 2 <= self.feature_dsm_ground_cells <= 64:
            raise ValueError('feature_dsm_ground_cells 必须为 2~64 的整数')
        self._number('feature_below_base_tolerance_m', self.feature_below_base_tolerance_m)
        if self.feature_below_base_tolerance_m < 0:
            raise ValueError('feature_below_base_tolerance_m 不能为负')
        for name in ('feature_height_vis_max_m', 'feature_center_radius_m',
                     'feature_step_height_m', 'feature_step_run_m'):
            self._number(name, getattr(self, name), positive=True)
        self._number('feature_min_class_confidence', self.feature_min_class_confidence)
        if not 0 <= self.feature_min_class_confidence <= 1:
            raise ValueError('feature_min_class_confidence 必须在 [0, 1]')
        if type(self.road_max_node_degree) is not int or self.road_max_node_degree < 3:
            raise ValueError('road_max_node_degree 必须为 >= 3 的整数')
        bands = self.feature_dsm_height_bands_m
        if not isinstance(bands, (list, tuple)) or not bands or bands[0] <= 0:
            raise ValueError('feature_dsm_height_bands_m 至少一个边界（米），且首个边界 > 0')
        for value in bands:
            self._number('feature_dsm_height_bands_m', value)
        if any(a >= b for a, b in zip(bands, bands[1:])):
            raise ValueError('feature_dsm_height_bands_m 必须严格递增')
