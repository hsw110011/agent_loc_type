# 卫星地图沿路预处理（global_blocks 整图流程）

本子包是项目主入口：整张卫星图 → MaGRoad 道路拓扑 → 沿路 75 m 块 → 每块语义/高度指标与退化判定。
使用项目 `.venv`、LangGraph `TypedDict` state、节点增量返回和逐节点 JSON 快照；保留源 RGB 像素网格，
支持普通 RGB 图片和 GeoTIFF，全局坐标使用输入图的像素坐标。`semantic_features`、`elevation_features`、
`vlm_candidate_detection` 仅有接口和占位状态。

旧的“切片 → 道路 mask（PathFinder）→ 骨架 → 采样点 → 窗口”流程及 PathFinder 已删除（2026-11-04），
只保留本整图流程；旧的版本 1 manifest 不再支持。

## 运行

在项目根目录执行：

```bash
bash agent/run_agent.sh --image input_geo_img/kitti_1003_0034_dom.tif \
  --config agent/map_preprocess/pipeline.yaml

# 同目录的 DSM 可自动配对，也可显式传入路径
bash agent/run_agent.sh --image input_geo_img/miluo_dom.tif \
  --dsm input_geo_img/miluo_dsm.tif --config <配置>

.venv/bin/python -m unittest discover -s agent/map_preprocess/tests -v
```

输出按输入图片名归档为 `<output_dir>/<image_name>/<task_id>/`，`task_id` 为“运行时间 + 6 位随机后缀”（如 `20261104_051230_3eb8a1`），按名字排序即运行先后，重复运行不覆盖。
每次运行在 `state.json` 和 manifest 中记录流程算法版本 `pipeline_version`（`global_config.PIPELINE_VERSION`）。会改变结果的算法修正都要递增它；
不同版本的结果不能直接比较，旧结果保留在原目录（版本 2：DSM 0 点只取语义路面、Node 5 裁剪整图语义、Node 3 面状区域改用距离变换；版本 3：高度分段改为 0.5 / 2 m；版本 4：Node 5 去掉可靠性标记，final_summary 不再因此判为不完整）。
缓存（MaGRoad 道路图、整图语义、道路拓扑）先写临时文件并 `fsync` 落盘再改名，写入时断电不会留下截断的缓存；读到损坏的缓存会重新计算。
CLI 标准输出最后一行为 manifest 的绝对路径（`state/manifest.json`），前一行是中文最终结论。失败时写 `state/error.json` 并保留已完成节点快照。

**提前结束**：Node 5 没有筛出候选块时跳过 Node 6-9，Node 6 没有确认任何退化时跳过 Node 7-9。两种情况都直接汇总，`final_summary.early_exit` 记录阶段和原因。空路网、专题数据缺失和支路判断未知记入 `issues`。

最终结论 `final_summary`：`refresh_global_manifest` 汇总各节点已有计数（不改变任何判断），写入 manifest 和 `state.json` 顶层，下游读这里即可，不要只看区域数是否为 0。

| 字段 | 含义 |
| --- | --- |
| `conclusion` | `degraded`：Node 8 产出了融合区域；`no_degradation`：Node 5-9 全部跑完（或按上面的条件提前结束），所有块都判断成功且没有确认的退化；`incomplete`：没有融合区域，但有节点未运行 / 被跳过 / dry_run、VLM 请求失败（Node 6、Node 9）或无法确认、已确认退化却未融合——不能当作无退化。Node 7 请求失败时基准保留进入融合，不算不完整 |
| `complete` | `issues` 为空时为 true；`degraded` 也可能为 false（例如部分区域推荐失败、其他块未判断） |
| `message` | 一句中文结论，同时写入日志和 CLI 输出 |
| `counts` | 块数、候选块、复核块、确认退化块、不退化块、删除的不可行驶块、退化区域、推荐区域、需人工复核区域 |
| `method_counts` | Node 9 各定位方式的区域数 |
| `early_exit` | 提前结束时为 `{after, stage, reason, label}`（`reason` 为 `no_candidates` 或 `no_confirmed_degradation`），否则为 null；提前结束后，之后的节点不算未运行 |
| `issues` | 每项含 `stage`、`kind`（`stage_not_run`、`no_blocks`、`no_data`、`request_failed`、`unverified`、`invalid_patch`、`not_fused`）、中文 `label`、`count`、`ids`。数据缺失另附 `reasons_by_block` |

配置支持 JSON/YAML，未知字段与非法参数报错，省略字段使用 `GlobalPreprocessConfig` 默认值。
正式 `pipeline.yaml` 为 Node 1b、6、7、9 统一开启 `medium`，每次请求最多使用 2048 token 思考。各节点的生成上限含思考和最终 JSON。

```python
from agent.map_preprocess import load_config, run_pipeline, load_manifest, crop_block

state = run_pipeline('input_geo_img/kitti_0930_0033_dom.tif', load_config('agent/map_preprocess/pipeline.yaml'))
manifest = load_manifest(state['manifest_path'])
```

## 环境与依赖

NumPy、Pillow、LangGraph、networkx、shapely、PyYAML、torch（MaGRoad / 语义分割），rasterio（GeoTIFF / DSM），
依赖清单统一在根目录 `requirements.txt`。rasterio 可选：存在时按窗口读取 GeoTIFF 并保留 CRS/affine 与有效掩码；
缺失时降级为 Pillow，通过 zoom/角点或 fallback 计算分辨率。核心库缺失时明确报错，不静默替换。

## 分辨率优先级

1. **GeoTIFF**：有 CRS 和有效 affine 时优先使用，`resolution_source="geotiff"`。投影 CRS 按其米/英尺等单位转米，旋转像元使用两轴向量长度。经纬度 CRS 在图像中心纬度按 CRS 椭球的子午圈/卯酉圈曲率把度换成米，不要求 pyproj。两轴不等时取算术平均并在超过配置容差时 warning；具体两轴值、CRS、affine 和计算方法写入 `image.resolution_details`。这里是原图网格的标量近似，尤其大范围经纬度图和有投影尺度变形的地图不能当作精确地面测距。
2. **Web Mercator**：无可用地理信息时，配置实际地图的 `zoom` 和 `center_lat`，按 `156543.03392 × cos(lat) / 2^zoom` 计算，`resolution_source="web_mercator"`。
   - 有 `zoom` 且只有 `corner_lonlat` 时，四角平均纬度作为中心纬度。
   - 只有四角时，按 Web Mercator 投影的四边长度/影像宽高推算，在中心纬度换成地面米，取两轴平均。**假设**角点依次为左上、右上、右下、左下，图像是未额外缩放的 Web Mercator 地图；支持跨日期变更线的局部范围。
   - 配置示例：`zoom: 18`、`center_lat: 39.9`。角点示例：`corner_lonlat: [[116.0,40.0],[116.1,40.0],[116.1,39.9],[116.0,39.9]]`。经纬度顺序始终为 longitude/latitude。
3. **fallback**：仍不能确定则 warning，采用 `fallback_resolution_m_per_px`，`resolution_source="fallback"`。不会自动猜测图片比例尺。

分辨率同时写入 state、manifest 顶层与 `manifest.image`。所有米制长度以此标量换算（块尺寸、清洗长度阈值、去重采样间距等）；`*_px` 参数始终是像素。四角不能恢复任意变形图片的真实比例尺。

## 节点、state 与输出

配置 `pipeline.yaml`（`agent/run_agent.sh` 的默认配置，文件开头的 `run_until` 决定跑到哪个节点）（`workflow: global_blocks`）；Python 使用
`GlobalPreprocessConfig`（通用字段继承自 `config.py:PreprocessConfig`）。入口 `main.py`、`graph.py`、
`agent/run_agent.sh`；仅 `segmenter: magroad_scene`（Node 1b）及启用 Node 6、Node 7、Node 9 时调用 VLM。

```bash
bash agent/run_agent.sh --image input_geo_img/kitti_0930_0033_dom.tif \
  --config agent/map_preprocess/pipeline.yaml

# 读取上一步输出的 manifest 路径，渲染所有块、ID、保留边及区域边界
.venv/bin/python -m agent.map_preprocess.debug_overview \
  --manifest output/map_preprocess/global_blocks/<image_name>/<task_id>/state/manifest.json \
  --output output/map_preprocess/global_blocks/<image_name>/<task_id>/visualization/overview.png
```

本模式的 `tile_grid=3` 仅建立无缝、无重叠的 3×3 归属区域，不裁切 RGB；设为 0
则禁用，块 `tile_id=null`。边跨区域不被拆开。

| # | 节点 | 做什么 | 读取的 state | 返回的 state 增量 | 落盘（state/） | 图片（visualization/） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `load_image_and_resolution` | 读原图尺寸与地理信息，算 m/px，划 3×3 归属区域；有 DSM 时检查与 RGB 对齐 | `image_path`、`dsm_path`、`config` | `image`、`tiles`、`resolution_m_per_px`、`resolution_source` | — | — |
| 1b | `classify_scene_type`（`segmenter: magroad_scene`） | 本地 Qwen 只看整图缩略图判断结构化/非结构化，程序映射到 MaGRoad 权重（非结构化 `magroad_wild_checkpoint`，结构化 `magroad_urban_checkpoint`）；失败/不确定/低置信度用 `scene_type_fallback`；结束后关闭 VLM 服务 | `image_path`、`image`、`resolution_m_per_px` | `scene_type`（场景、来源、选用权重） | `scene_type.json`（输入、原始输出、决策） | `00_scene_type_inputs/` |
| 2 | `extract_road_graph` | 整图道路图：默认用 Node 1b 选出的 MaGRoad 权重（按输入哈希 + 权重/阈值缓存） | `image_path`、`image`、`resolution_m_per_px` | `raw_road_graph_path`、`road_graph_cache`、`input_sha256` | `raw_road_graph.json` | — |
| 3 | `clean_road_graph` | 掩码拓扑（MaGRoad 概率 + 语义分割生成路网，见“Node 3：掩码拓扑”）、外部约束、可选分支复核；`road_topo_enabled=false` 时回退旧的逐步清洗 | `fused_road_graph_path`（无补路时 `raw_road_graph_path`）、`image`、`resolution_m_per_px` | `clean_road_graph_path`、`clean_graph_overview_path` | `clean_road_graph.json`（保留边、删除边及原因、过滤统计） | `01_road_cleaning.png`；启用复核时 `branch_review_input.png` |
| 4 | `build_blocks_and_groups` | 沿每条边按 71.25 m 弧长步长布 75 m 块→保留道路覆盖的几何去重→块邻接与分组，写 manifest | `clean_road_graph_path`、`image`、`tiles`、`resolution_m_per_px` 等 | `manifest_path` | `manifest.json`（nodes/edges/blocks/groups/dedup_stats/files） | `02_road_topology.png`、`03_blocks_initial.png`、`04_blocks_final.png` |
| 5 | `compute_block_features`（`block_features_enabled`） | 计算块级语义及可选相对路面高度统计，按阈值筛选候选块并记录缺失数据 | `manifest_path` | `block_features`（摘要、可用和缺失数据） | `blocks[].features`、`blocks[].review_types_available`、`block_features.json` | `05_blocks_degraded.png`、`06_degraded_block_sheets/`、`07_per_block_images/` |
| 6 | `vlm_review_blocks`（`vlm_review_enabled`，需 Node 5） | 本地 Qwen 独立复核候选块命中的语义或高程支路，确认不可行驶的块删除 | `manifest_path` | `vlm_review`（删除、失败、未确认块） | `blocks[].vlm_review`、`degraded_scenes.json`、`vlm_review.json`、`vlm_review/`、`manifest_before_vlm_review.json` | `09_vlm_review.png`、`10_vlm_review_inputs/` |
| 7 | `vlm_expand_blocks`（`vlm_expansion_enabled`，需 Node 6） | 对 Node 6 确认退化的支路，中心不变，扩展到 100 m（`vlm_expansion_sizes_m`）判断退化是否延续，程序计算连续确认的最大已测试尺寸；semantic 与 height 独立记录，不改块和 Node 6 结论 | `manifest_path`、`vlm_review` | `vlm_expansion`（摘要） | `degradation_extents.json`、`vlm_expansion.json`、`vlm_expansion/`；`extensions.vlm_expansion` | `11_vlm_expansion.png`、`12_vlm_expansion_inputs/` |
| 8 | `fuse_degraded_patches`（`degradation_fusion_enabled`，需 Node 7） | 把 Node 7 每条完成支路的最大确认窗口作为 Patch，按正面积重叠求连通分组并计算真实并集；纯几何，不调用 VLM，不改前序结论 | `manifest_path`、`vlm_expansion` | `degradation_fusion`（摘要） | `degradation_regions.json`；`extensions.degradation_fusion` | `13_degradation_fusion.png` |
| 9 | `recommend_localization`（`vlm_localization_enabled`，需 Node 8） | 本地 Qwen 逐融合区域推荐一种辅助定位方式（gps / lidar_inertial_slam / wheel_imu_odometry）；模型不接触坐标，bbox 等由程序按区域 ID 补回；不改 Node 8 结果 | `manifest_path`、`degradation_fusion` | `localization_recommendation`（逐区域摘要） | `localization_recommendations.json`、`localization_recommendation/`；`extensions.localization_recommendation` | `14_localization_recommendation.png`、`15_localization_inputs/` |
| — | `refresh_global_manifest` | 把扩展节点摘要与全部节点快照路径同步到 manifest，并汇总最终结论 | `manifest_path` 及扩展字段 | `final_summary` | 更新 `manifest.json` | — |

state 只存路径和轻量元数据，不存 RGB、模型、networkx 图或块列表；块与指标都在 manifest 中。
`semantic_features`/`elevation_features`/`vlm_candidate_detection` 仍为占位节点（默认关闭），启用时只记录 `implemented=false`。

### 任务目录：状态与可视化分开存放

```
output/map_preprocess/global_blocks/<image_name>/<task_id>/
├── state/                         # 状态与数据（JSON/日志/表格）
│   ├── config.json  run.log  state.json（最后一个节点后的完整 state）
│   ├── node_states/01_load_image_and_resolution.json …  # 每个节点结束时的完整 state，编号即执行顺序
│   ├── raw_road_graph.json  fused_road_graph.json  clean_road_graph.json
│   ├── road_topo/                 # 掩码拓扑的 graph.pkl、network.geojson、stats.json
│   ├── manifest.json              # 下游唯一入口
│   ├── block_features.json        # Node 5 摘要（退化块 = 送大模型名单）
│   ├── block_features_table/      # export_feature_table 输出 features.csv / features.md
│   ├── degraded_scenes.json       # Node 6 主要输出：VLM 确认的退化场景清单
│   ├── vlm_review.json  vlm_review/  manifest_before_vlm_review.json  # Node 6 摘要、每次请求记录、复核前备份
│   ├── degradation_extents.json   # Node 7 主要输出：每条支路的最大确认尺寸、停止原因和各档记录
│   ├── vlm_expansion.json  vlm_expansion/  # Node 7 摘要、每次请求记录
│   ├── degradation_regions.json   # Node 8 主要输出：融合后的退化区域（真实并集、外接矩形、来源 Patch）
│   ├── localization_recommendations.json  # Node 9 主要输出：每个区域的辅助定位推荐
│   └── localization_recommendation/<region_id>.json  # Node 9 每次请求记录
└── visualization/                 # 全部图片，编号即建议查看顺序
    ├── 01_road_cleaning.png       # 清洗：保留边彩色、删除边红色
    ├── 01b_road_topo/             # 掩码拓扑每个阶段的图、summary.png、stages.png、overlay.png
    ├── 02_road_topology.png       # 最终道路拓扑：粗线、白点为路口（度 >= 3）、边 ID；白色描边 = 语义补全边
    ├── 03_blocks_initial.png      # 去重前：沿每条边独立布块
    ├── 04_blocks_final.png        # 最终全图块（同一条边同色）
    ├── 05_blocks_degraded.png     # 全图块上标出退化块（橙=语义，蓝=高度，红=两者）
    ├── 07_per_block_images/<block_id>/{rgb,semantic_color,semantic_overlay,dsm_raw,height,height_bands}.png
    ├── 06_degraded_block_sheets/ # 自动生成的退化块拼图
    ├── 08_contact_sheets/         # 手动抽块拼图（feature_contact_sheet）默认输出
    ├── 09_vlm_review.png          # Node 6：退化场景（绿）/不退化（白虚框）/不可行驶已删除（红叉）/未确认（黄）
    ├── 10_vlm_review_inputs/<block_id>/{local_rgb,local_semantic,context_semantic}.png（semantic）或 {context_rgb,local_rgb,local_dsm}.png（height）+ <type>_input.json + <type>_metadata.json  # 输入图、精简说明、本地元数据
    ├── 11_vlm_expansion.png       # Node 7：各支路最大确认尺寸（semantic 绿、height 橙，粗框=已扩展，黄=基准不完整）
    ├── 12_vlm_expansion_inputs/<block_id>/<type>/{rgb,<type>}.png + <type>_input.json + <type>_metadata.json  # 同范围双图及档位框
    ├── 13_degradation_fusion.png  # Node 8：并集按类型半透明填充（红=degradation，蓝=longitudinal_degradation），青框=来源 Patch，黄框=外接矩形
    ├── 14_localization_recommendation.png  # Node 9：按推荐方式着色（绿=gps，蓝=lidar_inertial_slam，橙=wheel_imu_odometry），黄框=需人工复核，灰=失败
    └── 15_localization_inputs/<region_id>/rgb.png + input.json + metadata.json  # Node 9 送入 VLM 的带边界 RGB 及本地元数据
```

目录名与文件名集中定义在 `output_layout.py`。旧任务目录（manifest 在任务目录根）仍可被各工具读取。

### 比例尺、推理与缓存

复用现有三级逻辑，原始像素网格不重投影：优先 GeoTIFF CRS+affine，经纬度在中心
纬度按椭球近似换算米；然后 Web Mercator；最后 warning 并使用
`fallback_resolution_m_per_px`（默认 1，仅是假设）。普通图片可配置：

```yaml
workflow: global_blocks
segmenter: magroad_scene
zoom: 18
center_lat: 40.0
# 或 corner_lonlat: [[左上经度, 左上纬度], [右上经度, 右上纬度],
#                   [右下经度, 右下纬度], [左下经度, 左下纬度]]
```

`RoadGraphExtractor.extract(image_path_or_array) -> RawRoadGraph` 协议在
`percection_tool/interfaces.py`，统一格式为 `vertices:[{node_id,xy}]`、
`edges:[{u,v,polyline_xy,confidence}]`、`metadata:{}`；坐标是源图全局像元边缘坐标，
折线首尾须对应节点，confidence 可为 null。`factory.create_graph_extractor()` 延迟选择
`magroad/graph_adapter.py` 或 `dummy.py`；不会导入未选的模型。

SAM 适配器对全图仅调用一次现有 `SAMRoadExtractor.predict_graph`，内部窗口融合、
有效像元掩码、设备、官方阈值、严格权重校验及 float32 保持不变；没有将 mask
骨架化或用几何交叉推导路口。`magroad_native.json` 保留原始节点/边/有向预测/分数/
推理元数据，通过 `array_files` 引用原始概率 `.npy`（无损保留浮点值）；PNG 仅作预览。
`max_tile_pixels` 在本模式限制整图读取像元数，超出报错，请根据内存显式设置。
Node 1、按需裁剪和 rasterio 总览支持窗口/降采样读取，但现有 SAM 接口仍需要全图 RGB
及有效掩码在内存中；本次没有改写网络或安装新依赖。

`cache_enabled=true`、`cache_dir=output/map_preprocess/cache`。缓存键包含输入文件
SHA256、分辨率、推理配置、完整 checkpoint SHA256（或 dummy 图文件 SHA256）。
命中后不推理；损坏 JSON 或缺失缓存附件会重新提取。缓存 JSON 原子替换，原始预测
保存在缓存目录（`<创建时间>_<图名>_<哈希前 12 位>`，旧的纯哈希目录仍能命中），任务 manifest 引用其路径。修改清洗/布块参数可复用推理结果。
自定义注入 `run_pipeline(..., extractor=...)` 时，只有提供稳定字符串 `cache_key`
才启用缓存；调用者需把实现及参数版本编码进 key，否则默认不缓存。

Node 3 掩码拓扑（`road_topo.run_arrays`：R/S 融合、骨架化、遮挡补全、面状区域、剪枝等）也缓存在 `<cache_dir>/road_topo/`（`nodes/road_topo_cache.py`）。
键为 MaGRoad 缓存键 + 整图语义缓存键 + 图像尺寸/分辨率 + 配置的 `road_topo` 一节 + `utils/road_topo.py` 源码哈希；
同一输入再次运行直接读出路网、每步统计和 `01b_road_topo/` 图（河西十几分钟 → 秒级），之后的清洗、语义路线和布块照常计算。
改 Node 4 及之后的参数仍命中；改 `road_topo` 参数或代码自动失效；修改 run_arrays 之前的输入准备要递增 `road_topo_cache.CACHE_VERSION`。
关闭 `cache_enabled`、Node 2 未缓存或注入的语义模型没有身份时不缓存。`filter_stats.road_topo.cache_hit` 记录是否命中，命中时 `steps` 的耗时为首次计算时的值。

离线模式显式设置 `segmenter: dummy`。`dummy_graph_path` 可指向上述统一图 JSON；
省略时生成一条带 degree-2 中点的合成直路，不代表真实道路识别。

### 清洗和分支复核配置

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `min_component_length_m` | 20 | 删除总边长不足阈值的分量 |
| `min_spur_length_m` | 5 | 迭代剪除连接端点的悬挂短边；孤立短路也按此阈值处理 |
| `parallel_merge_dist_m` / `parallel_max_angle_deg` / `parallel_min_overlap` | 18 / 20 / .5 | 并行去重（见“Node 3：道路拓扑优化”）；距离 0 关闭，保留高分边 |
| `road_connect_dist_m` | 3 | 悬空路端离别的路不超过该距离（米）就接上；0 关闭 |
| `short_road_max_m` / `small_loop_max_reach_m` | 10 / 32 | 一端悬空且短于前者（米）的边删除；离开路口的最远距离不到后者（米）的自环删除；0 关闭 |
| `road_through_only` | false | 贯通路网总开关（见“Node 3：道路拓扑优化”）；两份流程 YAML 为 true |
| `road_border_margin_m` / `road_close_node_dist_m` / `small_cycle_max_length_m` | 10 / 16 / 120 | 贴图边判定距离、路口最小间距、小环周长上限（米）；后两者为 0 关闭对应项 |
| `road_shared_min_length_m` | 0（两份 YAML 为 12） | 支路贴着干路走到路口的长度达到该值（米）时，在真实分岔处建节点，重合段只留一条；0 关闭 |
| `road_max_shared_gap_m` / `road_max_node_degree` | 0（两份 YAML 为 10）/ 4 | 节点路数上限：超过后者的路口拆成相距前者（米）的两个节点；前者为 0 关闭 |
| `road_skeleton_fill_min_length_m` | 0（两份 YAML 为 40） | 概率骨架补漏的最短路段（米）；0 关闭，需要语义路线的整图语义和模型概率图 |
| `road_paved_ring_min_road` / `road_paved_ring_max_perimeter_m` | 0（两份 YAML 为 0.6）/ 300 | 内部路面占比达到前者、周长不超过后者（米）的环按宽路口网格或分隔带双线处理；需要语义路线；前者为 0 关闭 |
| `road_duplicate_prob` / `road_duplicate_min_support` | 0.18 / 0.6 | 贯通路网内的重复路去弱：两条边首尾相同时，沿线模型道路概率达到前者的长度比例低于后者的边删除，最强的一条始终保留；没有概率图的分割器跳过；后者为 0 关闭 |
| `road_semantic_route_enabled` / `road_semantic_route_corridor_m` / `road_semantic_route_node_snap_m` | false / 15 / 8 | 语义路线开关、路线离原线的上限、路口吸附半径（米）；两份流程 YAML 开启，上限为 25 m |
| `road_smooth_sigma_m` | 6 | 曲率平滑的高斯 sigma（米），90° 转角磨成半径约 1.6×sigma 的圆弧；0 关闭；两份流程 YAML 为 10 |
| `score_length_scale_m` | 100 | 长度分数 `min(1, length/scale)` |
| `edge_score_weights` | length .25 / confidence .4 / straightness .2 / connectivity .15 | 非负权重，按权重和归一化 |
| `min_edge_score` | .2 | 删除分数不足阈值的边 |
| `keep_main_trunk_only` / `min_attached_edge_length_m` | false / 20 | 仅保留主干与足够长的单跳附加边 |
| `route_polyline` / `route_coordinate_system` | [] / pixel | 全局像素折线；lonlat 模式需要源图 CRS/affine 和 rasterio |
| `roi_mask` / `route_buffer_m` | null / 50 | 同尺寸非零区域 mask、约束缓冲距离（米）；多项约束取交集 |
| `constraint_sample_spacing_m` | 1 | ROI 沿边采样的最大物理间隔，额外限制至 0.5px；route 使用整条几何覆盖判断 |
| `branch_review_enabled` / `reviewer_type` / `review_top_n_edges` | false / pass_through / 10 | 可选 pass_through 或 rule_based；后者只保留得分前 N 条 |

假设：缺失置信度按 1 评分，压缩链的置信度为已知片段按长度加权平均；弯曲度分数为
端点距离/折线长度（闭环为 0）；连通性分数为 `min(1,(degree(u)+degree(v))/8)`。
并行去重前计算候选分数，平滑后按最终连接关系和几何重算。纯环保留确定性锚点与自环。

假设：“主干”定义为总边长最长分量的最大生成树加权直径，附加边须连接该直径上的
节点且长度达标；环路的非树边同样按附加边规则处理。外部约束先于最低评分筛选，
不在缓冲区内的边优先记录 `external_constraint`，约束内仍须满足评分条件。
route 必须整条落在缓冲区；ROI 以 <=0.5px 采样检查，每个样点必须在非零区域或其
欧氏像素缓冲半径内，因此 ROI 判定是离散近似。ROI 读取受 `max_tile_pixels` 保护。

`BranchReviewer.review(image_path, annotated_image_path, edges)` 返回 `EdgeDecision`
列表，字段为 `edge_id,is_main_road,road_type,keep,reason`。
`road_type` 只接受 main/secondary/local/non_road；缺失、重复、类型非法的决策 warning
并默认保留。可用 `run_pipeline(..., reviewer=...)` 注入实现。复核开启时才绘制候选边
细线/旁注图，无任何 VLM 调用。所有阶段丢边都保存完整边信息及 `reason`；过滤统计
包含各阶段前后的边数与总长度，复核拒绝另存 `review_reason`。

### 布块配置、几何和版本 2 manifest

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `block_size_m` / `block_overlap_ratio` | 75 / .05 | 方块物理边长和沿路名义重叠；默认步长 71.25 m |
| `blocks_per_group` / `group_overlap_blocks` | 3 / 1 | 连续组大小与共享块数；末组保留剩余实际块 |
| `long_edge_strategy` | split_continuous | 连续覆盖；sparse3 只取贴起点/中点/贴终点三块并 warning |
| `context_size_m` | 300 | 上下文方窗大小，null 关闭；只记录几何，不裁图 |
| `max_blocks_total` | null | 超预算按低分优先丢整边；允许清空到零，不改块间距 |
| `debug` / `debug_block_ids` | false / [] | 仅为显式指定块保存示例 PNG |
| `tile_grid` | 3 | 无重叠区域网格边长，0 关闭；不生成 Tile 图像 |

连续布块时，边长 L≤size 取弧长中点一个块；否则按固定步长 size×(1−ratio) 放置
n=ceil((L−size)/stride)+1 个块。各中心都在道路拓扑上，超出路段的范围平分给两端。
关闭去重时，边端点的块通过原拓扑节点连接其它边端块，
序列组不跨边。前驱/后继描述本边方向的端点邻接，不代表交通方向；相连的两个起点
可互为 predecessor。同一块覆盖短边两端时同时记录两端关联。

假设：节点沿用原始模型 node_id，压缩边按完整折线字典序规范方向、稳定编号（从 0
开始，不表示交通方向）。块严格使用 `E{edge_id}_B{k}`，组为 `E{edge_id}_G{k}`；
丢边后不重新编号。同一输入和配置重复运行边/块 ID 相同。

新 manifest 使用 `format_version=2, workflow=global_blocks`，避免改变历史版本 1
的 points/windows 字段含义。顶层包含 `image,resolution_m_per_px,resolution_source,
input_sha256,tiles,nodes,edges,blocks,groups,dropped_edges,filter_stats,coverage,
review_decisions,extensions,config,files,task_id`。`files.clean_graph` 是 Node 3 的预算前
图；manifest 的 edges/nodes 是预算后的最终布块图。

每块记录 `block_id,edge_id,seq,global_xy,global_bbox,crop_bbox,size_px,size_m,
valid_global_bbox,valid_window_bbox,padding_ltrb,arc_length_m,arc_length_px,tangent_xy,
tangent_angle_rad,node_ids,tile_id,predecessor_ids,successor_ids,features,context`。
`features` 始终为空；`context` 是相同 BBox/有效区域元数据或 null。
`global_bbox` 为精确物理尺寸的连续像元边缘坐标，中心不变，允许越界；
`crop_bbox` 为实际读取的整数半开框，边长 round(size_m/resolution)，至少 1px，
栅格对齐误差不超过半像素。有效区域与 padding 针对 crop_bbox，所有路径均相对项目根。

初始覆盖指标按沿边弧长上的 `[center−size/2,center+size/2]` 区间并集计算，作为方块
真实覆盖的保守下界。沿图像横轴的直路上，相邻框横向重叠 3.75 m；纵向直路同理。
斜路和弯道上的框仍对齐原图坐标轴，实际二维重叠随方向和曲率变化。连续布块覆盖率为 1，
长边 sparse3 则报告覆盖空洞。

```python
from agent.map_preprocess import (
    GlobalPreprocessConfig, run_pipeline, load_manifest,
    crop_block, crop_context, crop_blocks_for_group,
)

# 无权重离线例；真实地图不传 segmenter，默认 magroad_scene
state = run_pipeline('input_geo_img/example.png', GlobalPreprocessConfig(segmenter='dummy'))
manifest = load_manifest(state['manifest_path'])  # 同时兼容历史版本 1
block_id = manifest['blocks'][0]['block_id']
rgb = crop_block(manifest, block_id, pad_mode='edge')
context_rgb = crop_context(manifest, block_id)
sequence = crop_blocks_for_group(manifest, manifest['groups'][0]['group_id'])
```

默认不会批量裁剪/保存块或上下文，避免重叠窗口导致重复 RGB 存储；按需读取源图，
越界使用 edge/reflect/constant 填充，并可跨归属区域读取原图有效像素。
调试总览使用 `overview_max_side` 限制输出尺寸，rasterio 直接降采样；Pillow 缺省降级
可能解码全图，继续遵守既有显式像元上限及 Pillow 大图保护。

本次新增：全局配置/YAML、四节点实现、清洗/复核/布块/渲染工具、SAM 整图适配和
合成测试；扩展已有 graph/state/config loader/manifest loader、包导出、感知
interfaces/factory/dummy 和总览脚本。未修改网络 vendor、权重、VLM、虚拟环境及依赖。

验证：现有 `.venv`（Python 3.11.16）下，47 项预处理测试（含新增 17 项全局
模式测试）和 12 项公共组件测试通过；`git diff --check` 及本次 20 个文件（含未跟踪
文件）的空白检查通过。统一启动脚本在程序生成的 1600×1200 GeoTIFF（0.5m/px）上
运行成功，产出 4 条边、14 个块、6 个组，覆盖率均为 1，按需块为 200×200 RGB，
上下文为 600×600 RGB，所有 manifest 文件引用可读取。运行证据在
`output/map_preprocess/global_smoke/verification.json`。这些是离线/模拟模型验证，
本次未运行真实 MaGRoad 权重或 VLM，不能据此宣称真实地图提取精度。

### 后续裁剪读取哪个文件

整图模式读取对应任务的 `manifest.json`：`blocks[].global_xy` 是裁剪中心 `[x,y]`，
单位为原图像素（不是经纬度）；`block_id` 用于传给 `crop_block(manifest, block_id)`。
`crop_bbox` 是实际整数读取范围，`global_bbox` 是物理窗口的连续像素范围；
建议使用 `crop_block()` 统一处理边界填充，避免自行按中心取整造成差异。
例如输入 `kitti_0930_0033_dom.tif`，后续输出目录为
`output/map_preprocess/global_blocks/kitti_0930_0033_dom/<task_id>/`。
块 ID 和 `manifest.json` 文件名保持不变，图片名体现在所属目录。

### Node 4：cross_edge_dedup（默认启用）

连续布块后、序列分组前，默认检查每个初始框的重复覆盖。

```yaml
dedup_enabled: true
dedup_method: walk          # walk（默认）或 greedy
dedup_prune_enabled: true   # 启用冗余剪除
dedup_prune_max_loss_m: 5   # greedy 的独占覆盖长度阈值
dedup_candidate_spacing_m: 10  # greedy 候选间距
road_sample_spacing_m: 1
min_road_coverage: 0.995    # greedy 停止条件
block_size_m: 75
block_overlap_ratio: 0.05
```

`walk` 复用初始框的中心和尺寸。一个框只有在与它各自重叠至少 50% 面积的保留框共同覆盖
其全部精确道路区间时才会删除，路尾与支路上短于采样间距的独占区间也保留。逐个检查删除，
优先保留长边上的框，同边优先保留靠前的框。`dedup_min_gain_m` 仅保留旧配置读取兼容性。

`greedy` 使用独立候选做全局贪心覆盖。启用剪除时，按最小独占采样长度逐个删除冗余框，
要求该长度低于 `dedup_prune_max_loss_m` 且删后采样覆盖率达到 `min_road_coverage`。
两种模式剪除的数量都记入 `dedup_stats.pruned_redundant_blocks`。

开启去重时 Node 4 输出 `visualization/03_blocks_initial.png`（去重前沿每条边独立布块）与
`visualization/04_blocks_final.png`（最终块），路径写入 `manifest.files`。

`dedup_enabled` 必须为布尔值；两种间距须为有限正数，`min_road_coverage` 范围为
(0,1]。
`dedup_enabled: false` 保留各边独立布出的框，并直接执行预算、邻接和分组。开启 `walk`
时，最终框是初始框的子集；`greedy` 另行选择中心。`dedup_prune_enabled: false` 可在
`walk` 中保留全部初始框，同时记录精确道路覆盖区间与去重统计。

greedy 的候选中心沿每条清洗边按 `dedup_candidate_spacing_m` 取样，包含首尾端点；道路覆盖
对象沿每条边按 `road_sample_spacing_m` 取样，也包含首尾端点。按原图连续全局坐标
判断点是否位于轴对齐方框内（包含边界）。假设共享路口点按所属边分别计数，
因此总覆盖率是“已覆盖逐边采样点数/全部逐边采样点数”，不保证每条边单独达到
该阈值，也不等同于精确道路总长度覆盖率。逐边采样和精确长度指标均保存供检查。

每次选择新增覆盖点数最多的块，直到总覆盖率达标或没有新增覆盖；平局按中心
`x,y,edge_id,arc_length_m` 升序打破。使用网格查询及惰性最大堆避免建立候选×道路点
的稠密矩阵，选择结果与逐轮全量重算一致。若候选间距太大导致无法达标，保留实际
结果并 warning，不伪造覆盖率。未选候选和采样点只存在于本节点内存，不写入产物。

保留块的 `edge_id` 是中心所在候选边，`seq` 按该边内中心弧长重新从 0 编号，ID
仍严格为 `E{edge_id}_B{k}`。同坐标候选的中心归属同样由上述平局规则确定。
BBox、上下文、裁剪、区域归属及空 features 契约保持不变。

新增 `blocks[].covered_segments` 为 JSON 三元组数组：
`[[edge_id, arc_start_m, arc_end_m], ...]`。它来自每条原始折线段与该方框的精确
裁切，按边及弧长排序并合并相接区间，不使用采样点近似区间边界。回返/自交折线的
不同弧长位置分别保留；只接触方框的点可产生零长度区间。汇总同一条边的所有区间
再取并集，即可还原实际被选块覆盖的道路区间。

前驱/后继清空后按 covered_segments 重建：同边区间按弧长连接相邻块；跨边仅在
原模型共享拓扑节点处连接覆盖端点的块，不会因二维道路相交新建道路图路口。
一块可覆盖多边，故其邻接可能来自多条边，不应把 predecessor/successor 当作交通
方向。`edges[].block_ids` 仍是中心归属该边的块，另加 `covering_block_ids` 表示
覆盖该边的全部块（按第一次覆盖弧长排序、同块只出现一次）。某条边没有中心归属块
时仍保留道路图与覆盖记录，不将其视为丢边。

分组规则保持不变：仅按块的中心归属边及中心弧长顺序重新分组，不跨归属边合组。
因此没有中心归属块的边不产生独立组，下游可用 covered_segments 或 covering_block_ids
找到覆盖它的邻边块。原 `blocks_per_group/group_overlap_blocks` 继续生效。

启用时新增 `manifest.dedup_stats`：

| 字段 | 含义 |
| --- | --- |
| `enabled` | true |
| `blocks_before` / `blocks_after` | 初始布块数 / 最终保留块数 |
| `candidate_blocks` | `walk` 的初始框总数或 `greedy` 的候选总数 |
| `pruned_redundant_blocks` | 剪除的冗余块数量 |
| `road_sample_count` / `covered_road_sample_count` | 逐边道路采样点总数 / 被覆盖点数 |
| `road_coverage` / `min_road_coverage` | 实际点覆盖率 / 本次阈值 |
| `mean_coverage_count_before` / `mean_coverage_count` | 去重前/后每个道路采样点平均被覆盖次数，分母包含未覆盖点 |

`coverage[]` 额外记录 `sample_coverage_ratio`、精确区间并集 `covered_intervals_m`、
中心归属块数 `owned_block_count`；`coverage_ratio` 此时是区间并集长度/原边长度，
`block_count` 是覆盖该边的块数，可能含中心归属其它边的块；此时名义相邻重叠率设为 null。
无道路时全局覆盖率约定为 1、平均覆盖次数为 0。

块预算在去重后检查，避免按旧块数提前删边；仍超预算则低分优先丢弃整条边，重做
候选选择和覆盖/邻接/分组，不随机丢块。此时 dedup_stats 和覆盖率分母针对预算后
保留的道路，预算丢边仍完整记录于 dropped_edges/filter_stats，不以缩小分母隐瞒丢边。
去重只增加 Node 4 的几何计算，不更改 MaGRoad 推理、模型图、缓存键或比例尺来源；
修改去重参数可复用既有原始预测。

本次验证：56 项预处理测试（新增 9 项去重测试）、12 项公共组件测试通过，
`git diff --check` 通过。新增测试核对十字路口/平行街道块数下降、采样覆盖阈值、
几何区间还原、输入顺序变化下的 ID 稳定、贪心平局、回返/切点、米/像素换算、
开关关闭时原字段一致，以及预算/全 LangGraph/按需裁剪集成。

复用已有 KITTI 真实图运行 `1c5c6694909f4a5784e834684b04fdc1` 的清洗图，仅重跑
Node 4（未重新推理模型），新产物为
`output/map_preprocess/global_blocks/kitti_raw_0930_0027/5163992d637b474ba20a7b9eb8f4e852/`。
100m 块由 85 个减至 45 个，677 个候选中选取 45 个；5968 个逐边道路采样点覆盖
5946 个（0.9963136729），平均覆盖次数由 2.6910187668 减至 1.4872654155。
61 条道路边全部保留，精确弧长区间并集/道路总长为 0.9964309841；阈值是全局
采样覆盖目标，个别边可低于 0.995。验证记录为该目录 `dedup_verification.json`。
独立框图位于 `output/debug_bbox/kitti_raw_0930_0027/5163992d637b474ba20a7b9eb8f4e852/`，
不接入主流程；原 85 块产物与调试图保持原样。

独立深度/用户语义分割适配器已提供于 `percection_tool`（见其 README），可通过
`python -m agent.run_perception --manifest ... --block-id ... --tool both` 按需分析块；
它们不自动接入本图，本文 semantic/elevation 扩展节点仍保持占位行为。

## 只跑到某个节点

打开 `pipeline.yaml`，开头列出了所有 `run_until` 选项，只保留一行不注释，流程就跑到该节点为止；全部注释等于跑完整流程。保留多行不注释会报错。停在 Node 4 之前时没有 manifest，终端打印的是任务目录。

## 运行日志（`run_logging.py`）

`agent/run_agent.sh` 运行时终端显示节点进度、耗时、CPU 温度和内存峰值。Node 6、7、9 的进度条按请求状态显示成功数与失败数。正常结束且通过 JSON 校验的响应即使没有 token 用量也计为成功。成功请求的用量和耗时可用时，进度条显示累计输出 token、整批输出吞吐和每路平均速度。运行日志也记录有用量及耗时的单次请求速度，耗时包含输入处理和生成。

每块覆盖率、过滤统计等明细只写 `state/run.log`。每个节点的用时写入 `state/node_timing.json`，中途失败也保留已完成的节点。

## Node 3：掩码拓扑（`utils/road_topo.py`）

`pipeline.yaml` 开启 `road_topo_enabled`，路网直接由掩码生成，不再使用 MaGRoad 的稀疏图，也不再运行下面几节的旧清洗（语义路线、骨架补漏、贯通路网、`min_edge_score` 过滤都不生效）。全部参数在 `pipeline.yaml` 的 `road_topo` 一节（米制），每一步可单独开关。

1. **输入**：`road_probability.npy` 线性上采样到原图后按 `road_thresh`（0.18）二值化得到 R；整图语义的道路类得到 S。需要真实的道路提取模型，dummy 分割器没有概率图，开启时报错。
2. **融合**：core = R∩S；S 独有的部分丢弃；R 独有的连通块只有桥接 core 或细长地延长主干时才并入。缝隙被植被占据且方向对得上时补路，建筑上一律不补。
3. **骨架转图**，迭代剔除毛刺、短的 filled 分支、面状区域内部分支、小环、孤立碎片。
4. **路口**：合并近距节点（隔着建筑不合并）、合并窄长环（隔离带造成的双线）、双节点合成十字路口、主路拉直并把支路接到主路上；三臂互不共线的真 Y 形不动。
5. **补断口**：路端互连（30 m 内，或方向偏差 10° 内时放宽到 80 m）、路端接主路（30 m 内），都不穿过建筑。连接后旧端点并入边，补上的长度记在 `filled_m`。
6. **形状**：近似直线的边拉直，其余 Chaikin 圆滑，最后拟合成“直线段 + 圆角”，相对原路径偏差不超过 3 m。

边除了原有字段，还有 `road_topo_source`（core/filled）、`width_m`、`filled_m`；节点有 `node_type`（junction/endpoint）。`dropped_edges` 里被删的分支 reason=`road_topo`。`edge_score` 只用于块预算排序。节点度超过 `road_max_node_degree` 时仍由 `split_crowded_junctions` 拆分。

单独运行（项目根目录，`.venv`）：`python -m agent.map_preprocess.utils.road_topo --road road.png --sem sem.png --gsd 0.05 --out out_dir`。合成场景在 `tests/road_topo_scene.py`。

## Node 3：过近路口合并（close_junctions）

以下几节是旧流程（`road_topo_enabled=false`）。

两端都是路口（度 >= 3）且短于 `junction_merge_dist_m` 的边，两端路口合并为一个，
该短边删除（`dropped_edges` 中 reason=`close_junction_merge`），其余边改接到新路口，最后重新 degree-2 压缩。
路端不被拉动，而是沿簇内置信度最高的路线（低置信的假连接不走）走到落点，原有路形和拐角保留；路线往回折时，从路上离落点最近处直接接过去。
落点取簇内路线的加权中位路口（路端走的总距离最短，同样近时取度最高、MaGRoad 边最多的），不取均值：汨罗合并前 115 个路口有 111 个在路面上，
旧做法（32 m、取均值）合并后 61 个里有 11 个落到路外。
默认阈值 = `block_size_m/4`（64 m 块为 16 m，约一个路口的宽度）；0 关闭。
按边长由短到长合并，且合并组内任意两个原始路口相距须 <= 阈值（节点记录 `merged_from_xy`，多轮合并也不累积），
不会把一串相邻路口压成一个点。

## Node 3：道路拓扑优化（`utils/road_refine.py`）

在过近路口合并之后按下列顺序执行，每步都可用配置关闭。连接关系仍以模型图为准：只有悬空路端会被接到别的边上，
两条边的几何交叉不建路口。`filter_stats` 记录各步前后的边数，`connect_roads` 另记接上的路端数与缺口，
`smooth_roads` 另记平滑前后最大转角与最大偏移（含切掉毛刺的距离）。

1. **并行去重**（`parallel_merge`）：沿边 2 m 采样，点到另一条边距离 <= `parallel_merge_dist_m`、局部方向（前后 5 m 的弦）
   夹角 <= `parallel_max_angle_deg`、且垂足不落在对方端点上时算并行。至少 `parallel_min_overlap` 比例的长度并行、
   整条边不离开对方 2 倍距离时视为重复（`dropped_edges` 中 reason=`parallel_duplicate`，`kept_edge_id` 为保留边）；
   互相覆盖时删低分边。被删边端点上还挂着别的边时，这些边改接到保留边上，路网不会被断开。
   改接可能形成新的过近路口，随后再跑一次合并（`parallel_close_junctions`）。
2. **近距连通**（`connect_roads`）：度 1 的路端离别的边 <= `road_connect_dist_m` 时，折线延长到目标边，投影点不在边端时打断目标边，
   不新增边。沿路网绕到目标点不足 10 倍距离的不连（已经连着，再连只造小环）。
3. **短路删除**（`short_roads`，reason=`short_road`）：一端悬空且短于 `short_road_max_m`（10 m）的边删除；挂在路口上、
   离开路口的最远距离不到 `small_loop_max_reach_m`（32 m，半个块）的自环删除。两端都接路口的短边是路网的一部分，不删。
   先连后删，能接上的短路不算悬空。
4. **贯通路网**（`enforce_through_network`，`road_through_only`，两份 YAML 开启、Python 默认关闭）：每条路都要有进有出。循环执行到不再变化：
   逐层删断头路（端点度 1 且离图边超过 `road_border_margin_m`，贴图边的路端算通向图外，reason=`dead_end_road`）；
   合并相隔不足 `road_close_node_dist_m` 的路口（reason=`close_junction_merge`，合并组直径上限为阈值的 2 倍）；
   自环不论长度一律删除（只连着一个路口，没有出口，reason=`self_loop`）；
   两条边首尾相同时，沿线模型道路概率支持不足的边删除，最强的一条始终保留（缓存目录的 `road_probability.npy`，语义补全边不参与，reason=`weak_duplicate`，`road_support` 记录支持度）；
   打开周长不足 `small_cycle_max_length_m` 的小环（重边、多点环，每次去掉环上两端能继续直行的端数最少的边，端数相同时去掉置信度最低的边，主路不被低置信度的短段打断，reason=`small_cycle`）；
   只留总长最大的连通分量（reason=`disconnected_road`）；贴图边路端到路口短于 `road_close_node_dist_m` 的短边删除（reason=`short_border_stub`）。评分过滤后再执行一次，这一次不合并路口。
   **路面环**（`collapse_paved_rings`，`filter_stats.paved_rings`，语义路线后再查一次）：贯通路网定稿后，环内部语义道路占比达到 `road_paved_ring_min_road` 时是宽路口网格或分隔带双线，不是街区（河西路口网格 0.8~1，KITTI 街区最高 0.42）：紧凑度 4πA/P² 不低于 0.4 的环收成一个路口节点，细长的环和两条边的环去掉一条边（reason=`paved_ring`）。两端都是路口、长度不足 40 m、中点周围 25 m 内路面占比达到 80% 的短边，两端合并到中点（`merge_wide_junctions`，reason=`wide_junction_merge`；河西 0.91 以上，KITTI 最高 0.52）。
   **分岔处建节点**（`split_shared_stretches`，`filter_stats.split_shared`）：从同一节点出发的两条边相距不超过 4 m 走了 `road_shared_min_length_m` 以上时，在真实分岔处打断置信度高的边并建节点，另一条从这里接入，重合段只留一条。
   **语义路线**（`road_semantic_route_enabled`，两份流程 YAML 开启，`filter_stats.semantic_route`）：贯通路网定稿后，先取整图语义（流程语义模型，默认 SegEarth-OV3，按输入哈希缓存），路线改走 road 类中线，连接关系不变。路口吸附到路面中心（半径 `road_semantic_route_node_snap_m`），路端不动；每条边在原线两侧 `road_semantic_route_corridor_m` 的走廊内求最小代价路径，没有 road 类的路段（树荫等）被拉回原线，建筑代价最高；绕路超过 1.35 倍或首尾相同的重复边保持原形。 路线代价按当地半宽归一（30 m 窗口内离路边的最大距离），宽路窄路中线的吸引力一样；宽路上路口吸附半径随半宽放大，最多 15 m。路线走完后再做一次分岔建节点（`filter_stats.split_shared_after_route`），分岔点周围 25 m 内路面占比达到 80% 的宽路口除外。语义路线只改路线形状，不增删连接。有模型概率图时（`road_probability.npy`）路线优先沿概率脊线走（概率 >= 0.3 处按当地最大概率归一），概率弱或断开处用语义 road 类补，路口也优先吸附到概率脊线，`filter_stats.semantic_route.on_probability_after` 记录线落在概率带内的比例。
   **骨架补漏**（`road_skeleton.fill_from_skeleton`，`road_skeleton_fill_min_length_m`，`filter_stats.skeleton_fill`）：贯通路网定稿后，MaGRoad 仍是主路网，只把概率掩码（概率 >= 0.3，语义 road 类只填概率带 12 m 内的断口）骨架上离路网超过 18 m、长度达到该值、概率强（沿线 70% 以上）、沿线建筑不超过 30%、两端离路网不超过 25 m 的路段补回路网。
   **节点路数上限**（`split_crowded_junctions`，`road_max_node_degree`，`road_max_shared_gap_m`，`filter_stats.crowded_junctions`）：平滑前，短于间距的边先合并，超过上限的路口拆成相距该间距的两个节点，最直的一对路作干路，其余支路按朝向分到两侧。
5. **曲率平滑**（`smooth_roads`）：拓扑定稿后才做。每条边去掉小于 sigma/6 的抖动和折返毛刺（单顶点转角超过 120° 且短臂不超过 3×sigma，更长的 V 形只做圆角）。
   路口处把偏折不超过 60° 且最顺的两条边配对，连成整条路线后沿弧长高斯平滑，路线穿过的节点保持原位，偏移沿弧长余弦衰减，所以切向在节点处连续；
   真实拐角（前后各 6 m 内走向变化超过 45°）保持原样、不做圆滑，平滑只作用在拐角之间的路段；未配对的支路在节点处转折。自环仍单独平滑。sigma 不超过路线总长的 1/4；急弯平滑后仍有尖点时再平滑一遍（最多 3 遍）。两份 YAML 的 `road_smooth_sigma_m` 为 12。
   `polyline_xy` 变为约 sigma/5 间距的密集点，`length_m` 随之更新，结果裁剪到图像范围内。

阈值依据四张真实图的融合图（河西、汨罗、KITTI 0930/1003）：有中央分隔带的同一条路被提成两条线时相距 11~18 m，
超过 20 m 开始误删 KITTI 与汨罗真实相邻的路。这四张图上的悬空路端离别的路都超过 7 m，3 m 连通不触发。

## Node 1b：VLM 整图场景判断 + MaGRoad（classify_scene_type，`segmenter: magroad_scene`）

配置 `pipeline.yaml`：

```bash
bash agent/run_agent.sh --image input_geo_img/kitti_0930_0033_dom.tif \
  --config agent/map_preprocess/pipeline.yaml
```

- 输入：整图缩略图（长边 `scene_vlm_image_px`=1024，无数据区涂黑；GeoTIFF 无掩码时大片纯黑视为无效），采样温度 `scene_vlm_temperature`=0。prompt 系统在 `agent/vlm/scene_type/`
  （prompt/skill/schema），模型先填路面、路网、建筑密度、地表覆盖四项观察，再给 `scene_type`（结构化/非结构化/不确定）及置信度。
- 决策（`agent.vlm.scene_type.decide_scene`）：非结构化 → `magroad_wild_checkpoint`（默认 `wildroad` 官方阈值）；
  结构化 → `magroad_urban_checkpoint`（默认 `globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455`）；请求失败、“不确定”或置信度“低”
  → `scene_type_fallback`（默认 `wild`）。`scene_type_override: wild|urban` 对所有图跳过 VLM；`scene_type_known`（图片文件名 → wild/urban）固定已确认的图，新图才调用 VLM。优先级 override > known > VLM。
- 权重写法同 `agent.run_magroad --checkpoints`。Node 2 调用 `percection_tool/magroad/graph_adapter.py`，输出统一道路图
  （边仍按旧约定记 `source=samroad`，表示“道路图模型边”）；原图超过 `max_tile_pixels` 时按权重训练 GSD
  降采样读取。缓存指纹含权重名、阈值覆盖与固定的权重 SHA256，换场景/阈值会重新提取。
- Node 1b 结束即关闭它启动的 VLM 服务，MaGRoad（WildRoad 峰值约 19.5 GiB）与 VLM 不同时占用显存。之后 Node 6 首次请求时再启动，
  Node 7、Node 9 复用同一服务，流程结束才关闭（`graph.shared_vlm_client`）；只启用一个 VLM 节点、关闭缓存或注入了分割模型时，各节点仍自行启动、关闭。
- 注入 `extractor` 时不加 Node 1b。

## Node 5：块级语义与 DSM 高程指标（compute_block_features）

当前按 DSM 是否可用自动选择退化分析分支。手动选择语义、高程或两项分析的命令见根目录 [拟新增的分析模式](../../README.md#语义与高程分析模式拟新增)。

输入两种情况，由是否提供 DSM 决定：

| 输入 | 道路拓扑 / 块 | Node 6 语义支路 | Node 6 高程支路 |
| --- | --- | --- | --- |
| 只有 RGB | RGB（MaGRoad） | 命中语义阈值才送审 | 无 DSM |
| RGB + DSM（同名配对或 `--dsm`） | RGB（MaGRoad） | 只记录语义占比 | 命中高程阈值才送审 |

```bash
bash agent/run_agent.sh --image input_geo_img/miluo_dom.tif \
  --dsm input_geo_img/miluo_dsm.tif --config <block_features_enabled: true 的配置>
```

DSM 不参与拓扑，Node 1 只检查它能否与 RGB 对齐并写入 manifest `image.dsm`（`utils/dsm.py`）：

- `geo`：两者都有 CRS/地理变换，CRS 与分辨率可以不同；地理范围必须有重叠；
- `pixel`：两者都没有地理信息时按像素对应，尺寸必须一致；一方有、一方没有则报错。

Node 5 按块的 RGB 像素范围换算地理范围，把 DSM 双线性重投影到块网格（RGB 网格按
`block_grid_factor`≈DSM 一个像元覆盖的 RGB 像元数下采样，不把粗 DSM 插值到 RGB 原分辨率）。
DSM 的 nodata、非有限值及绝对值 > 1e4 的值记为无效。语义标签最近邻缩放到同一网格后参与可靠性检查。

只处理 manifest 中每块方块在原图内的部分（读取 `valid_global_bbox`），超出原图的部分直接裁掉、不补边；
300m `context` 不参与。每块只看自身场景：不与邻块比较、不做全图排名。语义分割全部块→缓存→卸载，
再逐块读 DSM、计算指标并输出图片。Depth Anything 不再参与 Node 5。

```bash
# 对已有 manifest 单独运行；--dsm 可给已有的只 RGB 结果补上高程（语义命中缓存）
.venv/bin/python -m agent.map_preprocess.run_block_features \
  --manifest output/map_preprocess/global_blocks/<image_name>/<task_id>/state/manifest.json [--dsm dsm.tif]

# 按需拼图：--degraded 全部退化块；--count/--seed 随机抽块；--block-ids 指定块
.venv/bin/python -m agent.map_preprocess.feature_contact_sheet --manifest <manifest.json> --count 10 --seed 0

# 导出指标中英文定义、分布统计和逐块数值（state/block_features_table/features.csv、features.md）
.venv/bin/python -m agent.map_preprocess.export_feature_table --manifest <manifest.json>
```

Python：`run_pipeline(image, config, dsm_path=..., semantic_segmenter=...)` 或
`run_block_features(manifest_path, config, dsm_path=..., semantic_segmenter=...)`；注入工具提供字符串
`cache_key` 才缓存。语义模型由 `perception_config`（默认 `agent/perception_tools.yaml`）的 `pipeline_semantic` 选择，默认 `segearth_ov3`。

### 产物

| 位置 | 内容 |
| --- | --- |
| `manifest.json` → `image.dsm` | DSM 路径、对齐方式、CRS、分辨率（m/px）、重叠比例、块网格倍数；无 DSM 时不存在 |
| `manifest.json` → `blocks[].features` | 下表指标 |
| `manifest.json` → `blocks[].feature_vis` | 该块图片路径；关闭可视化时为 null |
| `visualization/07_per_block_images/<block_id>/` | `rgb`、`semantic_color`、`semantic_overlay`；有 DSM 时再加 `height`、`height_bands` |
| `visualization/05_blocks_degraded.png` | 全图块上标出退化块（按语义/高度/两者着色） |
| `visualization/06_degraded_block_sheets/` | 自动分页汇总退化块。每块展示 RGB 原图、语义分割图，有 DSM 时再展示相对路面高度图和高度分段图；最右栏汇总语义比例、高程分段比例和初筛结果 |
| `visualization/08_contact_sheets/` | 按需运行拼图工具的输出 |
| `state/block_features.json` | 摘要：`height_source`、DSM 元数据、整图语义缓存命中、各阈值诊断列表及可评估块、缺失数据和路面基准列表 |
| `manifest.json` → `blocks[].review_types_available` | 记录可用的专题数据 |

语义不再逐块推理：Node 5 直接从 Node 3 的整图语义（`cache_dir/semantic_whole/`，`mask` 0 road/1 building/2 plant/255 无效 + `confidence`）
按块的 `valid_global_bbox` 裁剪（`semantic_crop`），与语义路线、Node 6/7 用同一份结果；修改阈值、分段或 DSM 无需重新推理。
摘要记录 `semantic_source=whole_image`、`semantic_cache_hit`、`semantic_working_size`。`feature_cache_enabled`/`feature_cache_dir` 只为兼容旧配置保留，不再使用。
图片默认缩到 `feature_vis_max_side=1000`；semantic 图例：road 红、building 蓝、plant 绿、other 品红、
无效深灰；height 为块内去偏相对高度（0 m → `feature_height_vis_max_m`=15 m，近似 viridis）；
height_bands 按分段离散着色，与 `height_band_pct` 一致。

### 指标（`blocks[].features`）

中英文名称与定义统一维护在 `feature_definitions.py`（`METRICS`）。Node 5 只输出指标和规则判定结果，不做可靠性标记。

| 字段 | 含义 |
| --- | --- |
| `class_pct` | 整块有效像元中 road/building/plant/other 占比（%，和为 100）；other 为最高类置信度低于 `feature_min_class_confidence` 的像元；SegEarth-OV3 只输出三类，置信度下限为 0.5，默认配置下 other 恒为 0 |
| `height_band_pct` | DSM 相对块中心路面高度的分段占比（%，只统计 0 m 及以上），`0-0.5m` / `0.5-2m` / `>2m`；无 DSM 为 null |
| `height_stats` | `base_ref`/`base_dsm_m`/`base_offset_m`（路面 0 点、来源及离块中心距离）、`below_base_pct`（低于路面不计入）、`terrain_slope_pct`（仅去坡时）、`median_m`、`p95_m`、`max_m`、`nodata_pct`；无 DSM 为 null |
| `degradation` | 退化判定，见下节 |

每块 DSM 以**块中心路面**为 0 点（`utils/block_features.py:dsm_height_features`）：

1. 块中心 `global_xy` 是布块时道路拓扑线上的点（块以它为中心裁剪），但拓扑线可能偏离实际路面（例如落在路边建筑上）。
   中心像元是 road 且 DSM 有效时，取中心 5 m × 5 m 内有效 road 像元的 DSM 中位数（`base_ref=center_road`）。
   中心不是有效路面时，取同中心 10 m × 10 m 内有效 road 像元的中位数（`base_ref=expanded_road`），
   最近入选像元的距离记入 `base_offset_m`。仍无有效 road 时，高程统计为空，不进入高度退化候选。
   台阶连通区域也从这些路面像元出发。
2. 相对高度 = DSM − 0 点，单位米、不缩放，与 DSM 高程基准和所在海拔无关。
3. 低于路面（< −`feature_below_base_tolerance_m`，0.3 m）的像元按**高度变化是否连续**分两类：
   - 台阶 = 水平 `feature_step_run_m`（1 m）内高差 >= `feature_step_height_m`（0.5 m）的直接阶跃；坡度连续变化不是台阶。
   - **缓坡低于**（`slope_below_pct`）：从块中心路面出发不越过台阶即可连通到的低处，视为正常地面，计为 0 m（计入 `0-0.5m`）。
   - **阶跃低于**（`step_below_pct`）：隔着台阶才能到达的低处（路堤边坡下、桥下、坎下、沟里），为不合理的低，不参与分段。
   - `below_base_pct` = 两者合计（仅供参考）。[-0.3, 0) m 视为路面 DSM 噪声，计为 0 m。
4. 分段 `0-0.5m` 路面/地面、`0.5-2m` 低矮物（车辆、灌木、矮墙）、`>2m` 建筑、树、围墙或高架（`feature_dsm_height_bands_m=(0.5, 2)`；高于 0.5 m 视为可用于高程匹配）。

下坡一侧由连续性规则处理（缓坡低于计为地面）；上坡一侧仍按相对高度落入高段。可选 `feature_dsm_detrend: true`（默认关）
先减去块内拟合的地形平面。

### 退化判定（`features.degradation`）

比较均为严格不等（恰好等于阈值不算退化）：

| 字段 | 条件 |
| --- | --- |
| `semantic_degraded` | 仅无 DSM 时判断：道路 < `feature_semantic_max_road_pct`（10%）且 植被 > `feature_semantic_min_plant_pct`（65%），即植被覆盖、几乎无路。有 DSM 时为 null |
| `height_degraded` | 有 DSM 时任一满足即为 true：① `0-0.5m` 段 > `feature_height_flat_min_pct`（90%），整块几乎和路面一样平；② **阶跃低于**路面的像元 `step_below_pct` > `feature_height_below_max_pct`（70%），如路堤、高架，四周隔着台阶比路低；缓坡低于不算。`>2m`（及 `0.5-2m`）占比大不算退化。无 DSM 为 null |
| `degraded` | 有 DSM 时取 `height_degraded` 的布尔结果，无 DSM 时取 `semantic_degraded` 的布尔结果，供 Node 5 诊断和可视化 |

同时记录 `height_reasons`（`flat`/`below_road`）、`height_flat_pct`、`height_below_pct`、`semantic_road_pct`/`semantic_plant_pct`。
`block_features.json` 的 `degraded_blocks`、`semantic_degraded_blocks`、`height_degraded_blocks` 保存阈值标记。
实际送审块记在 `review_candidate_block_ids`，各块缺失的专题数据记在 `unavailable_review_data`。
有效 DSM 缺路面参考时，`missing_road_reference_block_ids` 记录块 ID，汇总标记结果不完整。
版本 4 起不再输出 `reliability`、`*_unreliable_blocks` 和 `*_dominant`，相关配置 `feature_max_other_pct`、`feature_min_mean_confidence`、
`feature_max_dsm_nodata_pct`、`feature_max_road_dsm_iqr_m`、`feature_min_class_pct` 已删除（配置里写了会报未知字段）。


## Node 6：退化块 VLM 复核（vlm_review_blocks）

本地 Qwen 独立评估 Node 5 送来的区域。语义侧关注类别占比、道路边界和其它类别侵入道路；
高程侧关注图示起伏与整体高度区分度，优先高出结构。本阶段不判断雷达实际可见性，也不决定定位方式切换。
Node 6 按 Node 5 的阈值结果分流，只为命中支路生成输入图并发起请求。整批只有高程候选时，Node 6 不读取整图语义。

| 分流条件 | 系统 | 输入 |
| --- | --- | --- |
| `semantic_degraded=true` | `agent/vlm/block_review/semantic/` | 局部 RGB + 局部语义 + 整体语义上下文（三图） |
| `height_degraded=true` | `agent/vlm/block_review/height/` | 整体 RGB 上下文 + 局部 RGB + 局部 DSM 伪彩色（三图） |

每套的 prompt 定义任务及图像含义、skill 给观察步骤、rule 给判据和程序映射、schema 定义输出。
两套系统不相互引用；模型只看到待评估区域，不收到 Node 5 的结论或触发原因。

Node 6 将退化解释为车端 BEV 与卫星侧地图缺少足够稳定、可区分的对应结构。语义匹配只使用道路、植被、建筑三类；
高程匹配使用车载激光雷达点云投影后每格的最大高度，对照卫星 DSM。两条支路都要求结合可见证据说明场景结构不足、沿路重复或俯视遮挡对匹配的影响。

### 输入与颜色

每次请求附带实际裁剪范围和图像来源。语义支路用局部 RGB 定位路带、用纯标签图对照并估计类别分布，
叠加上下文辅助追踪道路。高程支路先用上下文和局部 RGB 定位，再与相对 DSM 对照。黄框由程序添加，叠加色深浅不代表置信度。

- 每张图前有文字标签。局部图以块中心为中心、边长 `block_size_m`（75 m），统一裁去地图外部分，同尺寸、位置对应，可以是长方形。
- **semantic 复核三图**：

  | 图 | 文件 | 内容 | 覆盖范围 |
  | --- | --- | --- | --- |
  | 第 1 张：局部 RGB | `local_rgb.png` | 原始卫星 RGB，不叠加任何标记 | 目标 Patch |
  | 第 2 张：局部语义分割 | `local_semantic.png` | 道路、建筑、植被等类别的彩色语义分割图，不画中心线 | 与第 1 张完全相同 |
  | 第 3 张：整体语义上下文 | `context_semantic.png` | 原始 RGB + 半透明语义叠加（透明度 0.7：RGB 占 70%，语义色占 30%）+ 黄色目标 Patch 边界 | 100 m，至少覆盖整个局部 Patch，裁去地图外 |

  局部图没有十字，中心位置由图像说明给出（横向、纵向百分比）；上下文几何（`context_global_bbox`、`context_extent_m`、`context_patch_box_px`）记在 metadata 的 `context`。
- **height 复核三图**（`local_rgb.png` 与 semantic 共用）：

  | 图 | 文件 | 内容 | 覆盖范围 |
  | --- | --- | --- | --- |
  | 第 1 张：整体 RGB 上下文 | `context_rgb.png` | 原始 RGB + 黄色目标 Patch 边界；用于理解周围地形、道路延伸、桥梁及疑似隧道 | `vlm_review_height_context_m`（100 m，裁去地图外） |
  | 第 2 张：局部 RGB | `local_rgb.png` | 原始卫星 RGB，不叠加任何标记 | 目标 Patch（75 m） |
  | 第 3 张：局部 DSM 伪彩色 | `local_dsm.png` | DSM − H_ref 的 Viridis 连续色，无阴影、无标记，白色 = 无数据 | 与第 2 张完全相同 |

  着色：H_ref 为第 1 张范围（100 m）内有效 DSM 的第 `vlm_review_dsm_ref_percentile`（20）百分位，只是着色基准，不是路面高程；
  不用整图绝对高程范围，也不按块各自拉伸，所有块共用固定物理范围 `vlm_review_dsm_range_m`（[-2, 8] m 线性：-2 深紫、+0.5 蓝、+3 青绿、+5.5 绿、+8 黄；每米同样的颜色变化，既不按块拉伸放大，也不用过宽范围压缩。该范围按汨罗实测取：8 个高度退化块的 p1/p99 为 -1.8/+6.8 m，饱和约 1%；全部 63 块高于 +8 m 约 8%，主要是建筑和树冠），
  同一高差在任何块都是同一颜色，超出范围按端点显示。图像说明不给 H_ref 和米数，只说明 Viridis 色带的颜色含义（深紫色较低 → 深蓝色低至中等 → 蓝绿/青绿中等 → 绿色较高 → 亮黄色最高、白色无数据）；metadata 记录 `dsm_reference`、`dsm_color_range_m`、`dsm_color_legend`
  和 `height_stats`（`nodata_pct`、饱和比例 `below_range_pct` / `above_range_pct`、相对高差 `p5_m` / `p50_m` / `p95_m`）。饱和比例大时再调范围。
  上下文范围内没有有效 DSM 时 H_ref 为空、第 3 张全白，模型应判数据不可用。
- `<type>_input.json` 仅保存 `block_id`、`review_type` 和 `image_description`。模型收到区域标识、复核类型及自然语言图像说明，不再收到块信息 JSON。
  图像说明只是“图 + 介绍”：各图是什么、用途、实际覆盖宽高、中心位置、黄框含义，以及颜色和透明度代表的意思（semantic 上下文写明语义颜色占 30%、原始影像占 70%；height 写明 Viridis 色带各颜色代表的高低）；不发送统计、参考高程等数据。
- 语义占比、无数据比例、道路方向和来源、像素尺寸、比例尺、中心坐标和原图范围保存在 `<type>_metadata.json`，供本地追溯及校验，不发给模型。
  `image_size_px=[宽,高]`、`image_extent_m=[宽度米,高度米]`、`m_per_image_px=[横向米/像素,纵向米/像素]`、`image_global_bbox` 和 `center_image_xy` 均保留原有精确含义。
- 模型根据图像可见面积和空间分布独立判断语义占优及高度结构是否充分；不提供程序计算的类别/颜色占比，也不告知 Node 5 结论。
- 完整 Schema 仅通过 API 的 `output_schema` / `response_format` 传入，正文不再重复粘贴 Schema。两套 Rule、Skill 仍按原有方式全文发送。
- 本地 `class_pct` 为有效语义像元内四类占比，`semantic_nodata_pct` 为裁剪后图像无数据占比；`height_stats` 以实际裁剪区域为分母。
  数据不可用的程序防御检查仍使用本地元数据。无地图内像元时不生成补黑任务，该块按未复核保留；地图内部真实无效值不填造数据。
- 内部计算仍用统一分析网格，图片和统计统一裁去地图外部分。

### 复核 schema、置信度与三态结论

字段名保留英文，所有判断枚举使用中文；`review_type` 仍为路由标识 semantic/height，布尔值和 null 不变。
语义复核使用 schema v10，要求 `degradation_check.road_comparisons`（1–3 条逐段对照）及 `verdict`（结论与理由）。程序采用 VLM 的退化结论，通行性独立决定是否保留。语义扩展每档复用同一结构。高度复核使用 schema v11，要求 `height_check.height_comparisons` 逐项记录路面与路侧高差、沿路结构对照和俯视遮挡证据。高程同样采用模型的 `height_check.verdict`。文本字段不设长度上限。结构化生成约束字段和类型，程序再次检查格式；字段间矛盾不算请求失败；先处理置信度，再按 Rule 仅将矛盾的相关字段视为未知，其它可靠证据继续判定。
不以“先输出证据”代替正确性验证。

语义支路先根据 RGB 路面、路缘及遮挡前后的线索确定路带，再对照语义图同位置的道路标签。证据按「位置｜RGB 路带依据及可见性｜同位置语义道路状态」填写，区分可见路面错分和遮挡段道路标签缺失。Node 7 逐档核对新增环带，并结合整个当前窗口判定。

复核记录的 `derived.decision_trace` 保存选中的规则、条件的实际值与允许值、字段修正，以及可行驶性门槛。扩展结果按档保存同样的依据，环带全无数据触发覆盖时记录 `override`。

| 字段 | 内容 |
| --- | --- |
| `observations` | 每张输入图一个观察字段，顺序同图片：semantic 为 `rgb`、`semantic`、`context`；height 为 `context`、`rgb`、`height` |
| `drivable_check` | 证据、`area_type`（含“未知”）及 `area_type_confidence`、uncertainty；可行驶性由 `drivable_result` 从 `area_type` 推导，模型不输出 |
| `data_check` | 证据、`status`（可用/局部受限/不可用/不确定）、`data_confidence`、uncertainty |
| `degradation_check` | 证据、dominant_class（仅记录）、dominance、road_boundary、road_intrusion、road_geometry、longitudinal_structure，后五项各带 `<字段名>_confidence`，uncertainty |
| `height_check` | 证据、`height_comparisons`、relief（仅记录）、raised_structure、height_distinctiveness、road_geometry、longitudinal_structure，后四项各带 `<字段名>_confidence`，uncertainty |

两条支路输出 `longitudinal_comparisons` 和 `verdict.evidence_refs`。明确结论需要引用已记录的对照；纵向退化须引用两个不同沿路位置。专题图全部无数据、模型无法确认数据可用或必要引用缺失时，舍弃该候选，保留原输出和原因供排查。未确认区域不进入退化场景清单。

语义 v10 和高程 v11 将 `verdict.result` 映射为 `derived.degraded` 的 true/false/null，`reason_code` 为 `vlm_verdict`。模型的理由和逐段对照保存在请求记录及确认场景清单中。通行性按 `drivable_result` 独立处理，明确不可通行时删除该块。
高度 v8 与旧语义输出按有序 `degradation_decision` 计算三态结论；`candidate_reasons` 和 `matched_reasons` 分别记录候选和确认规则。两条支路都保留字段置信度和一致性诊断。
清单 `area_type` 从所有成功且支持最终通行结论的复核记录的 `effective_area_type` 中选取，包括未判退化的复核类型；类型不一致或无有效类型时填“未知”，不读取低置信度原始类型。

前置检查：数据不可用/不确定（`insufficient_data`）、`data_confidence=低`（`uncertain_data_quality`）给退化未知；
高程规则还有 `area_type=不可通行`（`not_drivable`，退化不适用）。**语义规则（`decision_ignores_area: true`）的退化结论不看地面类型，可不可通行只决定块的去留**；
被树荫遮挡的道路要求 VLM 依据遮挡两端、缝隙路面和路旁走向作判断，只有毫无线索时才用未知/不确定。之后按独立有效证据判定；局部矛盾不再作为全局前置阻断。
没有其它确定结论且存在字段冲突时以 `inconsistent_fields` 记录未知。下表是旧语义记录的规则优先级，语义 v9 由模型结合对照作答：

| 优先级 | 条件 | 结论 |
| --- | --- | --- |
| 1 | RGB 能辨认道路，语义图其它类别明显侵入、压窄、截断或覆盖有效路带 | 退化 |
| 2 | 单一语义类别大面积占优，同时道路语义边界混杂、破碎或消失 | 退化 |
| 3 | 道路呈明显直线，沿路语义类别、排列及空间结构高度重复 | 纵向退化 |
| 4 | 道路及周边语义结构清楚完整、没有侵入，并能排除纵向退化 | 不退化 |
| 5 | 不满足上述确定条件，或关键证据不足 | 未知 |

占比由 VLM 根据图像可见面积及布局独立判断，无固定建筑达标阈值或豁免。直道本身不足以判纵向退化，两侧类别相同也不足以证明沿路重复。
高程：比目标路面明显更高的结构（按色标换算，约 1 m 以内的差别不作依据）是明显高出结构。RGB 辅助辨认建筑、植被、山坡、高架桥，实际高度按同位置颜色读取。
Rule 只用物理含义描述高出/低处/无数据，颜色对应由图像说明给出；Node 6、Node 7 用同一 Viridis 着色，Node 7 复用同一份核心 Rule。
可通行区域周围明显高出结构只占整幅图一小点或分布稀少时，即使清楚也判退化；充分且清楚时继续检查纵向退化。
数据充分时，大片与路面同高的平地或主要为低处且缺少高出结构，也判退化；高低结构并存时按高出结构是否充分判断。
道路近似直线，且两侧高度分布沿道路方向持续相似、重复或几乎不变时，判纵向退化，即使高出结构充分。
只在一个横截面相似、或仅两侧颜色相同但沿路轮廓有变化，不足以确认纵向退化。高出结构清楚充分且能排除纵向退化时，才判不退化。
不设固定面积/距离门槛；直道或重复单独均不足以判纵向退化。`raised_structure` 表达清晰度，`height_distinctiveness` 表达周围高出结构是否充分，前者“清晰”可以同时对应后者“低”。
`raised_structure=无明显高出结构` 直接判退化；`较弱` 不能判不退化，分布不充分时判退化，否则未知。
低于参考的颜色只表示低于着色基准，并非相对于中心道路的固定高差；无数据优先进入未知。

`road_geometry` 的直线枚举在语义侧为“明显直线”，高程侧为“近似直线”，其余为“非直线／不适用／不确定”；`longitudinal_structure` 为“重复／有变化／不适用／不确定”。
非直线道路可排除本规则的直道纵向退化；已确认广场等非线状区域用“不适用”，看不清用“不确定”。
道路形态不确定但能确认沿路结构有变化时，也可排除重复；无法排除时不得仅凭边界清晰或黄红充分判正常。

一致性约束在 Rule 中以 `when`、`invalidate`、`reason` 表达，程序直接读取并在加载时校验字段及枚举。
先将低置信度值归一化；含未知/不确定的关系不构成确定矛盾。道路形态与沿路结构的适用性冲突时，仅屏蔽这两个字段；
高程侧“无明显高出结构”与“高度区分度高”冲突时，仅屏蔽这两个高度字段，沿路结构仍可单独支持纵向退化。
语义边界消失不等于侵入：只有 RGB 可辨道路且对应位置被其它类别明显覆盖等，才认定侵入。核对的是 RGB 真实路带与同位置标签，明确错分即使不在道路中心附近也算侵入；全为道路类的图可同时“边界消失”和“无侵入”，按单类占优规则判断。

每项置信度是模型自评，未作概率标定。`area_type_confidence=低` 时地面类型视为未知；其它支持“不确定”的判定字段“低”只替换该字段；通用替换明确排除 `area_type`、`data_status` 和仅记录字段，其余证据照常判定；
`data_confidence=低` 使结构判断未知。高度未知不会覆盖明确的 RGB 道路判断；数据不可用不等于不退化。

| 合并后的结果 | 处理 |
| --- | --- |
| 有明确不可行驶结论，且没有相反的明确道路结论 | 删除块及其相关引用 |
| 两套明确道路判断互相冲突；或高程复核的道路未知 | 保留，`degraded_scene=null`，不自动删除 |
| 至少一种退化成立（语义复核不要求确认可行驶，高程复核要求可行驶） | 进入 `degraded_scenes.json`，同时记录其它未确认类型 |
| 所有请求类型均明确不退化（同上） | 保留，`degraded_scene=false` |
| 没有确认退化，且还有未知或失败的类型 | 保留，`degraded_scene=null`，不混入不退化清单 |

`status=uncertain` 表示有有效输出但结论未定，`partial` 表示部分请求失败，全部失败为 `error`。
`unresolved_review_types` 列出未知或失败的类型，`drivable_conflict` 标记相反道路判断。
`unverified_block_ids` 收集整体场景结论未知的块。`unresolved_branch_block_ids` 记录支路未确认的块。已确认一种退化的块仍可进入场景清单，下游 `complete` 会标记判断不完整。
总览图：退化场景绿框、不退化白虚框、不可行驶已删除红叉、未确认黄框。

首次运行备份 `state/manifest_before_vlm_review.json`，重跑从备份开始；Node 5 重算前恢复备份。
历史输出不迁移、不按新 schema 重新解释；新语义复核使用 schema v10，新高程复核使用 v10。
`manifest.format_version=2`、块 ID、坐标及固定当前块范围保持不变。删除后的 coverage/dedup_stats 仍是复核前统计。

`recommended_size_m`、`expand`、`review_global_bbox` 仍固定为当前块；尺寸扩展由 Node 7 单独记录。
`reviews.<type>.coverage` 记录该请求实际送入模型的 `image_global_bbox` 与 `image_extent_m`，Node 7 据此核对基准。

### 运行

```bash
# 整条流程启用 block_features_enabled: true、vlm_review_enabled: true
.venv/bin/python -m agent.map_preprocess.run_vlm_review --manifest output/.../state/manifest.json [--block-ids E0_B0] [--dry-run]
```

`--dry-run` 只生成输入图、说明、元数据和提示词。`--block-ids` 对指定块运行 VLM。两者的产物都写入 `state/diagnostics/node6/` 和 `visualization/diagnostics/node6/`，正式产物由全量运行写入。
正式 `pipeline.yaml` 使用 `vlm_review_reasoning_effort: medium` 和 `vlm_review_thinking_budget: 2048`。预算为 0 时关闭思考，正整数为 token 上限，null 不限。单独运行 Node 6 时沿用 manifest 的档位和预算，也可通过 `--config` 指定。
其它配置包括 `vlm_review_enabled`、`vlm_review_image_px`（768，内部分析网格边长，越界裁剪后输出可能更小）、
`vlm_review_max_tokens`（每条 8192，含思考和 JSON；超出时该块记为失败、未复核保留；需比预算至少多 512）、`vlm_review_workers`（5；本地服务默认最多五路，单请求上下文上限 32768 token，五路共用 73728 token KV 缓存）、`vlm_review_dry_run`。

Node 6、Node 7 的批量推理在终端（stderr）显示 tqdm 进度条，按已完成的请求数推进并给出耗时和预计剩余时间；进度不写入 `run.log`。首个请求要等 VLM 服务启动，进度会先停在 0。
历史参数 `vlm_review_semantic_min_building_pct` 和 `vlm_review_grid_cells` 仍可读取，但自 v2 起不再用于决策或模型输入。
thinking 与服务生命周期保持原有实现，服务配置见 `agent/vlm/.env.example`。

## Node 7：退化区域尺寸扩展（vlm_expand_blocks）

对 Node 6 已确认并保留的退化区域，在中心点不变的前提下按尺寸档位逐步扩大观察范围，输出从基准开始连续确认仍退化的**最大已测试尺寸**。
semantic 和 height 两条支路独立评估、独立记录。本阶段不判断是否切换定位方式。达到 100 m 的含义是“最大已测试尺寸 100 m 仍退化”，不声明实际退化边界恰好为 100 m。

### 档位与进入条件

| 尺寸 | 含义 |
| --- | --- |
| 75 m（`block_size_m`，小 Patch） | 基准，继承 Node 6 的结论与置信度，模型不重复判断 |
| 100 m（`vlm_expansion_sizes_m`，大 Patch） | 扩展档，模型判断 |

支路须在 Node 6 中确认退化（`scene_types` 含该类型，即可行驶且 `degraded=true`），否则记 `skip_reason=node6_not_confirmed`（Node 6 未知、不退化或请求失败）。
基准范围不完整（Node 6 实际覆盖小于基准窗口，或窗口内缺支路所需数据）时不扩展、不调用 VLM，但保留 Node 6 的结论：`status=completed`、`max_size_m=75`、`stop_reason=base_incomplete`，缺失情况记在 `base_problem`，Node 8 用 Node 6 实际判断的范围作为 Patch。Node 6 原结果在每条记录的 `node6` 字段原样保留。
基准完整性同时核对 Node 6 的实际覆盖（`reviews.<type>.coverage`）和支路所需数据足迹，不凭配置尺寸推断。

### 地图边界（程序处理，不交给 VLM）

- 窗口须完整落在支路所需数据内：semantic 为 RGB 与语义，height 为 RGB 与 DSM。同一位置两条支路的有效档位可以不同。
- 足迹只把**与栅格边缘连通**的空白算作地图外：RGB 看掩码及纯黑/纯白填充（河西 DOM 未声明掩码，地图外是纯黑；KITTI 0930 有白边），语义看无效标签，DSM 看无效值。
- 地图内部的空洞不算越界，窗口仍有效，由模型按核心 Rule 的 `data` 部分检查；程序另记录各档环带无数据占比，环带无有效数据时按证据不足舍弃该档（`ring_no_data`）。
- 无效档位不画框、不进入 schema 和说明；配置多个扩展档时，遇到无效档后更大的档位也不评估；只有完整基准有效时不调用 VLM，`max_size_m=75`。

### 图像输入

每条支路一次请求、两张图：RGB，语义图或 DSM 伪彩色高程图。两图范围等于最大有效档位，只叠加相同的观察中心十字和档位框，不画道路拓扑线（Node 6 输入同样不画）。
框线是白芯黑边细线，按实际档位顺序使用基准实线、首个扩展档短虚线、后续档点线。按当前配置，75 m 是实线，100 m 是短虚线。框内左上角标尺寸，最外层框略向内收以保持可见。

- 语义图裁自 Node 6 同一份整图标签，各档不重新分割。
- 高度图沿用 Node 6 的参考高程和固定色带。按当前配置，减去块中心 100 m 范围内有效 DSM 的 P20，再按 [-2, 8] m 线性映射到 Viridis。所有扩展档共用该基准和色带，图像范围按最大有效档裁剪。
  图像说明的颜色含义与 Node 6 相同（深紫色较低 → 亮黄色最高，白色无数据）。参考高程、色标记在 metadata 的 `dsm_reference`、`dsm_color_range_m`、`dsm_color_legend`。
  100 m 范围内没有有效 DSM 时不调用 VLM，`stop_reason=base_not_comparable`，保留基准结论。

### 判断与停止

模型接收区域标识、`review_type`、两图用途与实际覆盖范围、颜色含义、框线与尺寸的对应关系、基准尺寸、有效档位和继承的地面类型。模型知道基准已经确认退化，Node 6 的具体退化类型、原因与证据留在本地 metadata。
不发送精确坐标、统计占比和处理参数，这些留在 `<type>_metadata.json`。

Node 7 的多尺寸 Rule 按支路说明怎样把 Node 6 的核心判据应用到扩大后的窗口。语义检查环带内的路带接续、同位置标签及整窗类别布局。高程检查新增结构的清晰度和分布、整窗高程区分度及沿路轮廓。判断时说明环带证据如何影响整个窗口的结论。关键对照无法完成时舍弃当前档，保留上一档已确认范围。

扩展输出 schema v4，高程每档复用三个对照问题。每档输出 `new_ring`（新增环带所见）、`data_check`、结构字段（与 Node 6 schema 完全相同）、`evidence`（一句话说明新增范围出现了什么及对当前窗口的影响）。
语义和高程各档采用模型的 `verdict`，每个明确结论都要引用该档环带证据 N 和窗口结构证据，最大扩展尺寸由程序根据各档结论计算。地面类型继承 Node 6 的 `effective_area_type`。

| 当前档 | 处理 |
| --- | --- |
| 退化或纵向退化 | 继续下一有效档 |
| 不退化 | 停止，保留上一档，`no_longer_degraded` |
| 未知（含关键字段低置信度、数据不足） | 停止，保留上一档，`insufficient_evidence`；不跳过未知档 |
| 到达最大档 | `max_tier_reached`；因边界没有更大档时为 `map_boundary` |
| 请求失败或输出不合 schema | 基准保留并进入 Node 8 融合，`request_failed`；`final_summary` 不算不完整，失败明细在 `state/vlm_expansion.json` 的 `errors` |

`walk_tiers` 由程序计算最大尺寸；模型在同一次输出中给出的停止档位之后的判断不采纳（只留在原始记录）。

### 输出

`state/degradation_extents.json` 的 `extents` 每条支路一条：`block_id`、`review_type`、`status`（completed/skipped）、`skip_reason`、`base_size_m`、`max_size_m`、`stop_reason`、`stop_label`、`valid_tiers_m`、`tiers`、`node6`。
`tiers` 每档含 `state`（退化/不退化/未知/未评估）、`source`（node6/vlm）、`degradation_type`、`reason_code`、`confirmed_reasons`、`evidence`、`field_confidence`、`data_status`、`data_confidence`、`uncertainty`。
未评估档位只有 `not_evaluated_reason`，不填造判断和置信度；未知不等于不退化。

块 ID、中心坐标、块尺寸、`blocks[].vlm_review` 和 `degraded_scenes.json` 不变，`manifest.format_version=2`。摘要写入 `extensions.vlm_expansion`，路径写入 `files.degradation_extents`、`files.vlm_expansion`、`files.vlm_expansion_overview`。
重跑 Node 7 覆盖自身产物；重跑 Node 5/Node 6 从备份恢复 manifest，旧的 Node 7 文件不再被 manifest 引用，需重跑 Node 7。

### 运行

```bash
# 整条流程启用 block_features_enabled、vlm_review_enabled、vlm_expansion_enabled
.venv/bin/python -m agent.map_preprocess.run_vlm_expansion --manifest output/.../state/manifest.json [--block-ids E0_B0] [--dry-run] [--resume]
```

`--dry-run` 只生成输入图、说明、元数据和提示词。`--block-ids` 对指定块运行 VLM。两者的产物都写入 `state/diagnostics/node7/` 和 `visualization/diagnostics/node7/`。
单独运行 Node 7 时，基准边长沿用 manifest 的 `block_size_m`，扩展档默认只有 100 m。历史任务记录的 64 m 基准会继续以 64 m 为基准。
`--resume` 只复用同流程版本、同请求指纹的成功结果。指纹包含图像文件内容、提示词、Schema 和推理参数。
正式重跑上游节点后，manifest 撤下过期的下游结果和快照引用，已汇总的任务同步更新 `final_summary` 和 `state.json`。
配置：`vlm_expansion_enabled`、`vlm_expansion_sizes_m`（正式配置和独立入口默认均为 [100]）、`vlm_expansion_image_px`（1024）、`vlm_expansion_reasoning_effort`（正式配置 medium）、`vlm_expansion_thinking_budget`（正式配置 2048）、`vlm_expansion_max_tokens`（12288）、`vlm_expansion_workers`（5）、`vlm_expansion_dry_run`。独立运行沿用 manifest 的档位和预算。高度着色沿用 Node 6 的 `vlm_review_height_context_m`、`vlm_review_dsm_ref_percentile`、`vlm_review_dsm_range_m`。

## Node 8：退化区域空间融合（fuse_degraded_patches）

只做确定性几何计算，不调用 VLM，不重新判断退化类型，不改 Node 6/7 的结论。几何规则在 `utils/patch_fusion.py`，读写产物在 `nodes/degradation_fusion.py`。

### 输入

读取 `manifest.files.degradation_extents`。每条 `status=completed` 且有 `max_size_m` 的支路是一个 Patch：

| Patch 字段 | 来源 |
| --- | --- |
| `patch_id` | `<block_id>_<review_type>`，如 `E0_B1_semantic` |
| `bbox` | `review_window(global_xy, max_size_m)`，与 Node 7 该档判断的窗口相同，全局像素；`stop_reason=base_incomplete` 时为 Node 6 实际判断的范围 `node6.coverage.image_global_bbox` |
| `degradation_type` | 最大确认档的结论：退化 → `degradation`，纵向退化 → `longitudinal_degradation`，不退化 → `non_degraded`，其余 → `unknown` |

`node6_not_confirmed` 的支路本来就不是退化，不构造窗口，记入 `not_degraded`（不算问题）；已确认退化却没有确认尺寸的支路记入 `excluded`，`final_summary` 列为不完整。Node 7 跳过或 dry-run 时本节点返回 `status=skipped`。

### 融合规则

1. 默认只处理 `degradation_fusion_types`（`degradation`、`longitudinal_degradation`），其余类别记入 `excluded`。BBox 非法（字段不全、非有限数值、零或负尺寸）或 `patch_id` 重复的记入 `rejected` 并写 warning，不中断流程。
2. 融合边只在两个**原始 Patch** 之间建立：同类型，且 `min(xmax)-max(xmin) > 0`、`min(ymax)-max(ymin) > 0`。共边、共点、有间隔都不连；不使用 IoU 阈值。`degradation_fusion_cross_type: true` 时不同类型之间也建边，这种组的 `degradation_type=mixed`。
3. 并查集求连通分量：A-B、B-C 重叠时 A、B、C 同组；孤立 Patch 保留为单成员区域。
4. 分组完成后才用 `unary_union` 计算并集和外接矩形，融合结果不参与任何重叠判断。

### 输出

`state/degradation_regions.json`：`regions`、`excluded`、`not_degraded`、`rejected`，以及 `coordinate_system=global_pixel`、`overlap_rule`、`fusion_types`、`cross_type`。`regions` 每项：

| 字段 | 含义 |
| --- | --- |
| `merged_id` | `Merged_1`、`Merged_2`…，按 `bbox` 排序，与输入顺序无关 |
| `degradation_type` / `source_degradation_types` | 区域类型 / 组内出现的类型 |
| `source_patch_ids` / `source_patch_count` / `source_patches` | 来源 Patch，`source_patches` 保留各自未改动的 `bbox`、`block_id`、`review_type`、`size_m` |
| `bbox` | 并集的最小轴对齐外接矩形，**仅用于裁剪、索引和可视化** |
| `union_polygon` | 真实并集，GeoJSON Geometry（`Polygon`）：L/T 形、环形的洞（第二个环）都保留。坐标是全局像素，不是经纬度 |
| `source_area_px2` / `union_area_px2` / `bbox_area_px2` | 融合前面积之和 / 并集面积 / 外接矩形面积 |

`union_polygon` 是 Node 7 已确认窗口的精确并集，`bbox` 中超出并集的部分未经验证；Node 7 的尺寸是最大已测试档位，窗口边界不代表实际退化边界恰好在此。摘要写入 `extensions.degradation_fusion`，路径写入 `files.degradation_regions`、`files.degradation_fusion_overview`（总览图 `visualization/13_degradation_fusion.png`）。重跑 Node 7 后需要重跑 Node 8。

配置：`degradation_fusion_enabled`（`pipeline.yaml` 开启，需 `vlm_expansion_enabled`）、`degradation_fusion_types`（只能取 `degradation`、`longitudinal_degradation` 的非空子集，其他值在读取配置时报错）、`degradation_fusion_cross_type`（false）。

## Node 9：辅助定位方式推荐（recommend_localization）

对 Node 8 的每个融合区域独立请求一次本地 Qwen，从 `gps`、`lidar_inertial_slam`、`wheel_imu_odometry` 中推荐一种。推荐只基于卫星影像的环境线索，不代表定位效果已验证。不重新判断退化，不改 Node 8 的结果，不发布 ROS。

Prompt 系统在 `agent/vlm/localization_recommendation/`：`node9_localization_recommendation_prompt.md`（角色与任务边界）、`skill.md`（分析步骤）、`rule.yaml`（三种方式的适用范围与选择规则）、`schema.json`（输出结构）。

### 选择规则

P0 同级比较 `lidar_inertial_slam`（lio_slam）和 `wheel_imu_odometry`（wheel_io）。前者需要路侧可能可观测、分布多样且沿路可区分的稳定几何。后者用于几何不足但有连续可通行路带的短段过渡。疑似隧道只采用可见的入口和接续线索，内部结构及通过时间留待验证。

P1 `gps` 只有在两种 P0 都不适用且路线沿途有持续开阔的证据时才考虑。根据近路建筑、树冠和覆盖物判断潜在遮挡及反射风险，实时信号需要在线验证。三种方式均缺少充分依据时给暂定项并标记人工复查。

### 输入

读取 `manifest.files.degradation_regions`，以 `merged_id` 为 `region_id`。每个区域按 Node 8 的 `bbox` 从原图裁剪（只限制在原图内，不外扩），保持长宽比缩放到长边不超过 `vlm_localization_image_px`。

每次请求只发送一张 RGB（`rgb.png`，黄芯黑边线标出 `union_polygon` 边界，区域内部不着色）、区域 ID 和尺度（实际覆盖宽高、米/图像像素），并说明这是上游识别出的潜在退化区域；区域正好占满整幅画面时（如单 Patch 区域），说明中明确没有线外部分。不发送退化类型或原因，不发送掩膜、语义图、高程、深度、GPS 状态或传感器参数。全局坐标、裁剪窗口和图像到全局的换算（`global = crop_global_bbox 原点 + image_xy × global_px_per_image_px`）留在 `metadata.json`。

### 程序保证

- 模型 Schema 没有 `bbox`、`degradation_type`、`union_polygon` 字段且禁止额外字段，`region_id` 按请求绑定为常量。`bbox`、`degradation_type`、`union_polygon`、`source_patch_ids` 按 `region_id` 从 Node 8 结果原样补回。
- Python 再次校验 Schema，并拒绝空白文本、未被否定的过度承诺用语（如“保证定位”“绝对可靠”，`rule.yaml` 的 `program.overclaim_phrases`）和可靠性数值（百分比、概率、评分，`program.reliability_number_patterns`）。三种方式只做定性比较。
- `assessment` 记录模型对两项 P0、沿途开阔程度、遮挡或反射线索的判断。只有两项 P0 为 `unsuitable`、`sky_view=open` 且 `obstruction_or_reflection=absent` 才接收 GPS。证据不足的 P0 暂定结果需标记验证和人工复查。
- 各方式的通用限制（GPS 需验证实时信号、LiDAR 需在线验证匹配约束、轮速 IMU 累积漂移和短时使用）由程序附加到 `limitations`。
- `wheel_imu_odometry` 或 `review_required=true` 时，`recommendation_status` 强制为 `requires_validation`。程序只收紧状态、只增加限制，不改模型选择的方式。

### 无效输出

JSON、Schema 或规则校验失败时，把错误信息反馈给模型重试；`VLMError`（服务瞬时不可用等）原样重试；次数由 `vlm_localization_max_retries` 控制（默认 1）。其他异常不重试。仍失败的区域记入 `failed_regions`（含 `error`，不含推荐字段），不影响其他区域，节点 `status=partial`。每个 Node 8 区域只出现在 `regions` 或 `failed_regions` 之一。Node 8 跳过时本节点返回 `status=skipped`；空区域列表写出 `regions: []`，不调用 VLM。

### 输出

`state/localization_recommendations.json` 示例（按 Node 8 的区域顺序）：

```json
{
  "coordinate_system": "global_pixel",
  "schema_version": 2,
  "regions": [
    {
      "region_id": "Merged_1",
      "degradation_type": "degradation",
      "bbox": [770, 9104, 2370, 10704],
      "recommended_method": "lidar_inertial_slam",
      "assessment": {"lidar_inertial_slam": "suitable", "wheel_imu_odometry": "uncertain", "sky_view": "obstructed", "obstruction_or_reflection": "present"},
      "reason": "道路两侧可见连续建筑与路缘，可能提供几何约束，且树冠使 GPS 存在遮挡风险。",
      "visual_evidence": ["道路两侧可见连续建筑", "存在清晰的路缘线"],
      "limitations": ["北侧树冠可能遮挡部分结构", "卫星影像只提供三维结构的间接线索，需确认车载 LiDAR 实际能观测到相应结构，并在线验证匹配约束是否充分"],
      "recommendation_status": "requires_validation",
      "review_required": false,
      "union_polygon": {"type": "Polygon", "coordinates": []},
      "source_patch_ids": ["E0_B1_semantic"]
    }
  ],
  "failed_regions": []
}
```

上面的取值只是结构示例。State 与 `extensions.localization_recommendation` 存摘要：`status`（completed/partial/dry_run）、`method_counts`、`regions`（不含几何轮廓和文本）、`failed_region_ids`、`errors`；路径写入 `files.localization_recommendations`、`files.localization_recommendation_overview`。重跑 Node 8 后需要重跑 Node 9。

配置：`vlm_localization_enabled`（`pipeline.yaml` 开启，需 `degradation_fusion_enabled`）、`vlm_localization_image_px`（1024）、`vlm_localization_reasoning_effort`（正式配置 medium）、`vlm_localization_thinking_budget`（正式配置 2048）、`vlm_localization_max_tokens`（4096）、`vlm_localization_workers`（2）、`vlm_localization_max_retries`（1）、`vlm_localization_dry_run`（只渲染输入和 prompt，不调用 VLM、不改 manifest）。

