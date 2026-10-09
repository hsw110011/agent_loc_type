# VLM LangGraph Agent

唯一图入口是 `langgraph_main.py`，唯一启动脚本是 `run_agent.sh`。默认流程为 `START → read_picture → extract_roads → END`。前两个节点不加载 VLM，也不读取两个 prompt 系统。

## 目录

```text
vlm/
├── langgraph_main.py       # 编排、CLI、节点间 state 快照
├── read_picture.py         # Node 1：读取和处理 TIFF
├── extract_roads.py        # Node 2：PathFinder 道路提取
├── candidates.py           # Node 3：identify_candidates
├── visualization.py        # Node 4：draw_candidates
├── parallel_vlm.py         # Node 5：assess_regions
├── client.py               # 节点共用的本机 VLM 调用
├── runtime.py              # 首次推理启动 INT8 服务，结束后关闭
├── common.py               # 节点共用的配置、校验、trace 和 state 保存
├── __init__.py             # 包入口
├── requirements.txt        # 所有节点和 VLM 的统一依赖
├── run_agent.sh            # 唯一启动脚本
├── PathFinder/             # 道路模型源码与 pathfinder.pth
├── node_flow.md            # 前四个节点输入输出流程图
├── candidate_area_sys/     # 候选识别 prompt 系统
├── loc_type_sys/           # 精细评估 prompt 系统
├── runtime/                # Node 3/4 的输入、配置、API 响应和 ROI 中间产物
├── log/                    # 按上海时间命名的运行与 VLM 日志
└── output/
    ├── <task_id>/          # 同一次任务的产物、trace、state
    └── verification/       # 验证记录及历史目录迁移记录
```

`client.py`、`runtime.py`、`common.py` 均被节点流程调用，需要保留。独立推理工具 `infer.py` 已删除。

## 一个环境、一个启动脚本

Agent、PathFinder 和 VLM 默认共用项目 `.venv-vllm/bin/python`，统一依赖位于 `requirements.txt`。当前环境已安装并通过依赖检查。

```bash
.venv-vllm/bin/python -m pip install -r vlm/requirements.txt

# 默认仅运行 Node 1/2
bash vlm/run_agent.sh --image /absolute/path/input.tif --primary-mode GPS

# 仅运行 Node 1；输入坐标顺序为经度、纬度
bash vlm/run_agent.sh --image /absolute/path/input.tif --primary-mode GPS \
  --stop-after input --geo-coordinates '{"longitude":110.0,"latitude":40.0}'

# 普通 TIFF 没有 CRS 时，需要提供原始米/像素
bash vlm/run_agent.sh --image /absolute/path/plain.tif --primary-mode GPS --gsd-m 0.5

# prompt 系统补全后再开启完整流程
bash vlm/run_agent.sh --image /absolute/path/input.tif --primary-mode GPS \
  --stop-after full --parallel-vlm
```

可通过 `PROJECT_PYTHON=/absolute/path/python` 指定另一个安装了全部依赖的环境。`.env.example` 只提供运行参数示例；启动脚本不自动读取项目已有 `.env`。`--output-dir` 可更换任务产物的根目录，默认是 `vlm/output`。

## Node 1：read_picture

验证 TIFF 扩展名、文件头和实际 GTiff 驱动。所有有 CRS 的输入真正重投影到 WGS84 三度带 Gauss，以影像中心经度确定分带和中央经线，比例因子 1、东坐标偏移 500000 米、北坐标偏移 0，东坐标不加带号，轴序为 Easting/Northing。

执行 `经纬度 → Gauss 米制坐标 → 像素列/行` 转换，在 Gauss 平面计算源像元尺寸，输出固定 `0.2 × 0.2 m/pixel`。双线性重采样保留波段和 dtype，无效像元 mask 使用最近邻。重新打开输出验证 TIFF、CRS、transform 和分辨率。普通 TIFF 没有 CRS 时使用 `--gsd-m` 建立本地网格，不能执行地理坐标转换。

state 的 `image_path` 指向处理后的 TIFF，源图另存 `source_image_path`。同时保存 RGB PNG、地理转换参数和 trace。默认输出上限 6400 万像元。uint8 RGB 保持原值，其它数值类型按有效像元的 2%～98% 分位数拉伸并记录范围。重采样不会增加源图的真实空间细节。

```python
from vlm.read_picture import lonlat_to_pixel, pixel_to_lonlat
point = lonlat_to_pixel(110.0, 40.0, state)
location = pixel_to_lonlat(100.5, 200.5, state)
```

像素坐标使用连续像元边缘坐标，像元中心需加 0.5。参数约定见 [PROJ](https://proj.org/en/stable/operations/projections/tmerc.html) 和 [Rasterio](https://rasterio.readthedocs.io/en/stable/topics/transforms.html)。

## Node 2：extract_roads

读取 `vlm/PathFinder/pathfinder.pth` 中的 `model_state_dict`，不恢复优化器。默认 512×512 分块、重叠 64 像素，重叠区平均道路概率，以 `>0.1` 输出背景 0 / 道路 1 的 mask，无效像元不标成道路。`auto` 优先 GPU，512 分块空闲显存不足 2 GiB 时选择 CPU并记录原因；可显式指定 `--pathfinder-device cuda:0`。

输出 mask 与处理后的 TIFF 保持尺寸、CRS 和 transform 对齐。增强图使用绿色半透明 mask 与黄色 polygon 边界（含孔洞）；完整像元边缘不简化、不删除小区域。有 CRS 时额外保存 WGS84 经纬度 GeoJSON。`--road-mask` 可提供已对齐的 mask，跳过模型推理。

## output 与 log

每个节点完成后保存 `state_<节点名>.json`，保留该节点传给下游的状态；`state.json` 保存最新状态，CLI 结束时更新为最终结果。状态快照不包含图片 base64，图片文件通过路径记录。节点失败时已完成节点的快照仍保留；trace 记录处理失败。

```text
output/<task_id>/
├── satellite_0p2m.tiff / satellite_rgb.png
├── coordinate_transform.json
├── road_mask.tiff / road_mask.png
├── road_enhanced.png / road_boundary.png
├── road_polygons.json / road_polygons.geojson
├── trace.json / trace.txt
├── state_read_picture.json
├── state_extract_roads.json
└── state.json

log/
├── YYYYMMDD_HHMMSS_<小数秒>_agent.log
└── YYYYMMDD_HHMMSS_<小数秒>_vlm.log
```

同一个 task_id 不覆盖已有目录。后续框图、ROI、评估输入包也保存在同一任务目录。旧 `reports` 的任务产物与验证记录已迁移到 `output`，旧日志迁移到 `log` 并改为时间命名；历史 JSON/TXT 中对应路径已更新。映射见 `output/verification/layout_migration.json`。

## VLM 延迟加载与 INT8

创建客户端、构图、编码图片和 Node 1/2 均不启动 VLM。第一次推理请求由 `runtime.py` 直接启动服务进程，在加载原始权重后使用 TorchAO `Int8WeightOnlyConfig()` 做 W8A16 量化。整个 `visual` 子树被排除，视觉编码器和连接层保留 BF16；原始权重文件不改写。依据 [vLLM TorchAO](https://docs.vllm.ai/en/v0.18.0/features/quantization/torchao/) 和 [PyTorch INT8](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int8WeightOnlyConfig.html)。

Node 3/5 共享一份模型，默认同时最多两个请求，其余排队；可用 `VLM_MAX_NUM_SEQS` 调整。GPU 内存预算默认 0.65，加载等待上限 600 秒。服务日志始终放在 `log`，不会写入任务产物目录。自动启动遇到端口占用会报错；`VLM_AUTOSTART=0` 可使用自行管理的外部服务。

CLI 成功或失败后都关闭自己启动的服务。Python 使用完整图时请管理客户端生命周期：

```python
from vlm.client import VLMClient
from vlm.langgraph_main import build_graph
with VLMClient() as client:
    state = build_graph(client=client, stop_after="full").invoke(input_state)
```

Node 3 的 candidate_area_sys 已检查并接通；Node 5 的 loc_type_sys 留待后续处理。Node 1/2 与 INT8 基础推理均已真实验证，记录在 `output/verification`；这些测试验证调用链，不代表分割准确率或量化后与 BF16 的精度一致。临时测试程序测试后删除。

## Node 3 的 Prompt 系统、加载位置与 Structured Outputs

Node 3 已采用 `candidate_area_sys/schema.json` 的新契约，不再使用旧 GPS 专用字段、归一化四角 bbox 或自动生成的 `area_001` 编号。模型结果只有 `areas`，每个候选包含 `area_id`、`bbox`、`scene_description`、`selection_reason`、`matched_rules`；ID 为 `Area_1`、`Area_2`…，空结果为 `{"areas": []}`。

`candidates.py:load_candidate_config` 的读取顺序为 **Prompt → Skill → Rule → Schema**。Prompt、Skill、Rule 按顺序拼入 API 的 system message；user message 中只有一张道路增强图、实际图片宽高、坐标约定及显式候选数量上限。Node 3 不传入或选择主定位方式，也不把经纬度或下游定位结论加入候选 JSON。

Schema 保留为 API `response_format={"type":"json_schema","json_schema":{"name":"vlm_output","schema":...,"strict":true}}`。程序额外根据实际送入图像的宽高约束 bbox 各顶点范围，并将 `--max-regions`（如果提供）设置为数组最大长度。服务明确使用 XGrammar，禁用后端失败自动回退；参考 [vLLM 0.17.1 Structured Outputs](https://docs.vllm.ai/en/v0.17.1/features/structured_outputs/)。返回后再用完整 JSON Schema 校验，并检查 ID 顺序、四边形顺时针/凸性/正面积/越界、区域两两内部互不重叠以及纯空白描述。业务校验失败时最多把错误反馈给模型纠正一次，每次请求、响应保存在 attempts/序号/，校验结果合并到 inference.json。仍不合格则保存 error.json 与失败 trace，不由程序改坐标；若第二次仅剩面积重叠，则按模型发现顺序保留不相交子集，连续编号并记录原始 ID 映射和排除原因；其他字段、规则引用或顶点错误仍失败。服务或传输错误不重试。

**VLM 的真正加载位置**仍是 `client.py:VLMClient._infer` 中的 `runtime.ensure_started()`。Node 1/2 不加载模型；Node 3 先完成配置读取、schema 校验和输入产物保存，再进入 API 请求并首次启动 INT8 服务。Node 4 只做 CPU 绘图和裁剪；完整 CLI 结束后关闭自己启动的服务。

模型输入保留长宽比，最多 `VLM_MAX_IMAGE_PIXELS` 像元；保存的 `vlm_input.png` 就是实际 API 输入图。Node 3 输出 bbox 使用这张图的像素边缘坐标，而不是内部 patch/token 坐标。bbox 现在为四个顺时针顶点 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`，优先表示贴合道路走向的倾斜矩形，也允许凸四边形；水平矩形同样使用四顶点。程序逐点按 x/y 比例映射到完整处理后图像，保留浮点坐标，不向外取整，以免引入重叠。Schema 约束格式，程序使用凸四边形分离轴检测保证内部不重叠，边界接触允许；不同倾斜框的水平外接框可以相交。Schema 和几何校验不能保证视觉判断正确；大幅面整图缩小后也不能保证细小道路候选召回。

道路 Mask 和 polygon 仅辅助搜索，不限制候选范围。Prompt、Skill、Rule 和实际调用指令统一采用宽泛道路定义：VLM 从原图判断为道路或可能可通行的区域均可纳入，包括未标注的土路、田间便道、场地通道、开阔硬化场地和连续裸露地表。潜在通行必须有视觉依据，并保留实际车辆通行条件尚未验证的不确定性。

```bash
# Node 1～3；无需提供主定位模式
bash vlm/run_agent.sh --image /absolute/path/input.tif --stop-after candidates

# Node 1～4；保存全部候选框可视化，不执行 Node 5
bash vlm/run_agent.sh --image /absolute/path/input.tif --stop-after visualization
```

## Node 4 的三类业务产物

Node 4 只在 `output/<task_id>/node4/` 保存三类文件：`regions.json`、`bbox_satellite.png`、各区域的 `Area_n.png`。不再保存独立的 Area_n.json、visualization.json、parallel_inputs.json 或预览图；state 与 trace 仍在任务输出根目录，API 调试记录仍在 runtime。

统一 `regions.json` 的 `areas` 中，每个区域包含五个候选字段和 ROI 路径、外接裁剪框、局部四边形、坐标偏移。公共的完整原图路径、全局框图路径、图像尺寸、坐标说明和几何校验规则只存一份。后续节点用 `region_json_path + area_id` 从该文件选择区域，不扫描目录、不读取独立区域 JSON。

全局框图为完整 0.2 m/pixel RGB 影像，只绘制候选框和编号，无道路 mask/polygon。ROI 内保留原图像素，区域外透明且 RGB 为零；ROI 坐标加 roi_offset_xy 对应全图坐标。

```text
output/<task_id>/
├── candidates.json / candidates_original.json    # Node 3 原始与全幅坐标候选
├── node4/
│   ├── regions.json                # 全部区域、matched_rules、路径与坐标说明
│   ├── bbox_satellite.png          # 全幅原始 RGB + 区域框/编号
│   ├── Area_1.png
│   └── Area_2.png                  # 按实际区域数量保存
└── state*.json / trace.json / trace.txt

runtime/<task_id>/node3/
├── prompt_system/                  # 本次使用的四份配置快照
├── vlm_input.png
├── context.json                    # 输入尺寸、配置顺序、实际 schema、坐标映射
├── inference.json                  # usage、校验记录、重叠排除与原始 ID 映射
└── attempts/01/                    # 有纠正时追加 02
    ├── request.json
    ├── response.json
    └── content.txt
```

## 规则追溯与互不重叠

每个候选的 `matched_rules` 为 `[{"rule_id": "...", "rule_name": "..."}]`，至少一项。程序从 Rule 的 candidate_patterns.id/title 和 selection_policy.id/title 编译成对的 JSON Schema 枚举；未知 ID、ID 与名称错配、重复引用会失败。规则含义仍由 VLM 结合视觉证据判断，ID 校验不能代替视觉正确性验证。通用筛选规则为不属于特定场景类型但有视觉依据的候选提供合法引用。

Node 3 和 Node 4 均逐对检查凸四边形内部不重叠，边界接触允许。第一次违规要求模型纠正；第二次若仍只有重叠，按发现顺序保留不相交子集，不移动顶点或修改规则依据。被排除区域和输出 ID 对应的原始 ID 保存在 inference.overlap_resolution，并传入统一 regions.json。此策略保证最终区域互不重叠，可能减少候选数量；不保证 VLM 第一轮就能全部选对。

regions.json 的 geometry_validation 记录程序已验证的 bbox_policy.id/title。空候选保存统一 JSON 与完整原图框图，areas 和下游任务列表为空。Node 5 的精细评估业务规则需要另行检查，不在本次验证范围内。

## 路径约定与流程图

所有 state、产物 JSON 和 trace 中的文件路径保存为相对于项目根目录 `hsw_lm/` 的 POSIX 相对路径，例如 `vlm/output/<task_id>/satellite_rgb.png`。加载统一调用 `common.resolve_path`，与启动时的工作目录无关；操作系统实际打开文件和启动服务时使用据此计算的绝对路径，代码不写死机器目录。绝对输入路径也兼容；外部输入保存为 `../...` 相对路径。项目内部目录整体移动后保持相对布局即可继续加载。

```bash
bash vlm/run_agent.sh --image baotou_rgb_0p2m.tif --stop-after visualization
```

PathFinder 源码与默认权重现在位于 `vlm/PathFinder/`。前四个节点的输入输出流程图和 Node 3 文件解释见 [node_flow.md](node_flow.md)。已有 output/runtime JSON 也已迁移为相对路径；历史 Node 3 调用记录已分组并移除重复副本，迁移记录保存在 `runtime/path_migration.json`。
