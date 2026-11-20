# 卫星地图退化区域识别与辅助定位推荐

输入一张卫星正射影像（可选配套 DSM），程序沿道路找出“卫星地图匹配容易失效”的退化区域，并为每个区域推荐一种辅助定位方式。
流程用 LangGraph 编排，道路、语义和深度等感知模型都在本地运行，退化判断与定位推荐由本地 VLM（Qwen3.8-27B）完成，全程不依赖外部服务。

## 流程

| 节点 | 做什么 |
| --- | --- |
| 1 读图 | 读取影像和地理信息，计算分辨率（m/px）；有 DSM 时检查与 RGB 是否对齐 |
| 1b 场景判断 | VLM 看整图缩略图，判断结构化道路还是非结构化道路，据此选择 MaGRoad 权重 |
| 2–3 道路拓扑 | 仅用 MaGRoad 整图道路概率生成拓扑，保留几何清洗与补接 |
| 4 布块 | 沿道路拓扑每 48.75 m 布一个 75 m × 75 m 小 Patch，预留 35% 沿路重叠；去除面积重叠至少 75%且无独占道路覆盖的块 |
| 5 块级指标 | 统计语义占比及相对路面高度分布，按阈值筛选候选块并记录缺失数据 |
| 6 VLM 复核 | 独立复核 Node 5 选中的候选块及基准无效的高程块，检查匹配结构、沿路重复与俯视遮挡 |
| 7 尺寸扩展 | 对已确认的退化块，中心不变，把范围扩大到 100 m × 100 m（大 Patch），用新增环带和整窗证据判断退化是否延续；证据不足时保留 75 m，确认仍退化时扩大到 100 m |
| 8 区域融合 | 同类型、有重叠的退化 Patch 按连通关系合并为退化区域；纯几何计算，不调用 VLM |
| 9 定位推荐 | VLM 看每个退化区域的影像，按 LiDAR-Inertial SLAM、Wheel-IMU Odometry、GPS 的顺序选择，GPS 仅在前两项均不适用且沿路持续开阔时考虑，并给出环境证据；不代表定位效果已经验证 |

Node 5 没有筛出候选块，或 Node 6 没有确认退化时，流程会提前汇总。空路网、缺数据和未确认的支路会记录为 `incomplete`，已有退化区域时则保留区域并标记 `complete=false`。

Node 5 按中心 5、10、20 m 方窗依次寻找路面基准，每档先取语义道路，失败后再用清洗道路掩码补选 DSM 样本。补选结果以 `base_ref=cleaned_road` 标记，语义图和语义初筛沿用原标签。基准默认要求至少 3 个有效道路像元、道路 DSM 有效率至少 50%、道路高程四分位距不超过 2 m。基准无效时留空道路相对高差和分段占比，记录原因，块内有有效 DSM 时直接送 Node 6 高程复核。

### Node 6 / 7 的 VLM 输入

模型只收到图片和图片介绍（每张图是什么、覆盖多大范围、颜色和透明度代表什么），不收到程序计算的统计数据，也不知道 Node 5 的初筛结论。输入图上不画道路拓扑线。Node 5 检出道路拓扑中心相对路面基准有向上阶跃时，额外送高程复核，并提示核查俯视遮挡、隧道或下穿入口，以及建筑占据道路中央的可能性。

| 复核 | 输入图（按顺序） |
| --- | --- |
| 语义（Node 6） | 局部 RGB（75 m，无标记）、局部语义分割图（同范围）、整体语义上下文（100 m，RGB 占 70%、语义色占 30%，黄框标出目标区域） |
| 高程（Node 6） | 整体 RGB 上下文（100 m，黄框标出目标区域）、局部 RGB（75 m，无标记）、局部 DSM 伪彩色高程图（同范围） |
| 尺寸扩展（Node 7） | 100 m 范围的 RGB 和语义图或 DSM 高程图，画有观察中心十字和 75 m、100 m 档位框 |

- 语义颜色：红色为道路或硬化地面，绿色为植被，蓝色为建筑，品红为其它或分类不确定，深灰为无数据。
- 高程着色：
  - 计算的是 DSM 减去参考高程（100 m 范围内有效 DSM 的第 20 百分位）的差值，按固定的 [-2, 8] m 线性映射到 Viridis 色带：深紫色最低，往上依次是深蓝、蓝绿、绿色，亮黄色最高；白色为无数据。
  - 所有块共用这一条色带，Node 6 和 Node 7 的着色方式相同。

## 目录

```
agent/                      LangGraph 流程、配置、state、VLM 客户端与 prompt 系统
├── run_agent.sh            启动脚本
├── map_preprocess/         主流程（节点、配置 pipeline.yaml、测试）
└── vlm/                    VLM 服务、客户端、各节点的 prompt / skill / rule / schema
percection_tool/            感知模型：magroad/、segearth_ov3/、depth_anything/（各自的推理代码、第三方源码和权重）
qwen3827b-exl3-dflash2/     VLM 权重与 ExLlamaV3 源码（独立 git，不在本仓库管理）
input_geo_img/              输入影像
output/                     运行产物（不入库）
tests/                      感知工具、VLM 服务和项目结构的测试
```

## 环境

- 硬件：单张 RTX 3090（24 GB）。感知模型和 VLM 分时使用显存，VLM 服务在需要时启动，用完即关闭。
- 软件：项目根目录下的 `.venv` 是唯一环境（Python 3.11，torch 2.10.0 + CUDA 12.8）。流程、感知模型和 VLM 服务共用这一个环境，依赖以 `requirements.txt` 为准。
- VLM 引擎 ExLlamaV3 需要从模型项目源码编译后安装进 `.venv`，步骤见 [agent/vlm/README.md](agent/vlm/README.md#环境)。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
# 然后按 agent/vlm/README.md 编译安装 ExLlamaV3；感知模型权重的下载见 percection_tool/README.md
```

## 输入

影像放在 `input_geo_img/`，格式为 GeoTIFF：

- `<name>_dom.tif` 会查找同目录的 `<name>_dsm.tif`。也支持 `.tiff`，显式 `--dsm` 优先。
- 无配对 DSM 且未指定 `--dsm` 时只评估语义支路。场景白名单先匹配原文件名，再匹配去掉 `_dom` 的文件名。

分辨率优先取 GeoTIFF 的地理信息；普通图片需要在配置里给出 zoom 和中心纬度，否则会使用带告警的默认值。

## 运行

在项目根目录执行：

```bash
# 只有 RGB
bash agent/run_agent.sh --image input_geo_img/kitti_1003_0027_dom.tif

# RGB + DSM
bash agent/run_agent.sh --image input_geo_img/miluo_dom.tif --dsm input_geo_img/miluo_dsm.tif
```

- 配置默认使用 `agent/map_preprocess/pipeline.yaml`，也可以用 `--config` 指定。文件开头的 `run_until` 决定流程跑到哪个节点为止，例如只跑到道路拓扑或 VLM 复核。
- 已有任务可以单独重跑某个节点，例如 Node 6 复核：

  ```bash
  .venv/bin/python -m agent.map_preprocess.run_vlm_review --manifest output/.../state/manifest.json [--dry-run]
  ```

  Node 6 和 Node 7 的 `--dry-run`、`--block-ids` 只写诊断目录。Node 6、7、9 正式重跑时暂存旧产物，若中途抛出异常则恢复旧产物和发布记录。新结果发布后，manifest 会撤下需要重新生成的下游引用。Node 5 用 `run_block_features` 重算，Node 7 用 `run_vlm_expansion` 重跑。

- VLM 进度条显示成功数、失败数和输出 token 速度。正常结束且通过 JSON 校验的响应计为成功，即使服务未返回 token 用量。

### 语义与高程分析模式

使用 `--analysis {auto,semantic,height,both}` 选择退化分析分支。命令行覆盖配置中的 `analysis`，配置默认使用 `auto`。

```bash
bash agent/run_agent.sh --image input_geo_img/hexi_dom.tif --analysis semantic
bash agent/run_agent.sh --image input_geo_img/hexi_dom.tif --analysis height
bash agent/run_agent.sh --image input_geo_img/hexi_dom.tif --analysis both
```

| 模式 | 分析范围 |
| --- | --- |
| `auto`（省略参数时） | 沿用当前逻辑，有 DSM 时分析高程，无 DSM 时分析语义 |
| `semantic` | 只分析语义，跳过 DSM 读取与高程复核 |
| `height` | 只分析高程，需要 DSM |
| `both` | 同时分析语义和高程，需要 DSM |

Node 5 按所选分支分别初筛。Node 6 为阈值命中的分支、基准无效且有有效 DSM 的块，以及道路中心向上阶跃的块生成对应输入；同一个块若两项都命中，分别用各自的图片和 prompt 发起两次 VLM 请求。Node 7 沿确认退化的分支判断能否扩大范围。DSM 可自动配对，也可用 `--dsm` 指定。

## 输出

每次运行的产物都放在 `output/map_preprocess/global_blocks/<图名>/<task_id>/` 下。`task_id` 由运行时间加随机后缀组成，重复运行不会覆盖之前的结果。

- `state/manifest.json`：下游的唯一入口，包含道路、块、各节点结果和最终结论。命令行输出的最后一行就是它的路径，倒数第二行是中文结论。
- `final_summary.conclusion`：最终结论，取值为以下三种之一：
  - `degraded`：发现了退化区域；
  - `no_degradation`：所有块都判断成功，且没有退化；
  - `incomplete`：有节点没跑完，或有块判断失败。这种情况不能当作“无退化”。
- 主要结果：
  - `degraded_scenes.json`：Node 6 确认的退化块；
  - `degradation_extents.json`：Node 7 的扩展尺寸；
  - `degradation_regions.json`：Node 8 融合后的区域；
  - `localization_recommendations.json`：Node 9 的定位推荐。
- `visualization/`：全部图片，编号就是建议的查看顺序。其中包括道路拓扑、块分布、各节点总览，以及每次送给 VLM 的输入图和图片介绍。

每次运行都会在 `state.json` 和 manifest 中记录流程算法版本 `pipeline_version`。版本号不同的结果之间，算法不一样，不能直接比较。

## 测试

```bash
.venv/bin/python -m unittest discover -s agent/map_preprocess/tests -v
.venv/bin/python -m unittest discover -s tests -v
```

离线测试用合成数据和假 VLM 运行，只能证明程序逻辑正确，不能证明真实模型的判断效果。

## 文档

| 文档 | 内容 |
| --- | --- |
| [agent/map_preprocess/README.md](agent/map_preprocess/README.md) | 各节点的算法、配置项、state 与产物细节 |
| [agent/vlm/README.md](agent/vlm/README.md) | VLM 服务的部署、环境变量与调用接口 |
| [percection_tool/README.md](percection_tool/README.md) | 感知模型说明与权重下载 |
| [AGENTS.md](AGENTS.md) | AI 助手修改本项目时遵守的约定 |

旧的 PathFinder 切片流程和 Qwen3.5-9B + vLLM 后端都已删除。
