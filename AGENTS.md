# AI 开发指南

本项目是卫星地图沿路退化区域识别与辅助定位推荐，主流程在 `agent/map_preprocess/`。开始前：

1. 运行 `git status --short` 并查看相关 diff，保留用户已有的修改。
2. 阅读根目录 `README.md`（整体流程）和 `agent/map_preprocess/README.md`（节点细节），再看目标节点的代码、state 和测试。

## 环境与运行

- 项目根目录 `.venv` 是唯一环境，流程、感知模型和 VLM 服务都用它，依赖以 `requirements.txt` 为准。不要新建其它环境。
- 启动脚本是 `agent/run_agent.sh`，Python 入口是 `agent/map_preprocess/main.py`，默认配置是 `agent/map_preprocess/pipeline.yaml`。
- 不要为了某次运行去修改 `pipeline.yaml`：需要不同参数时另写一份临时配置，用 `--config` 传入，放在 `output/` 下。
- 真实模型运行之前先核对资源：GPU 只有一张 RTX 3090，用 `nvidia-smi` 确认没有别的任务占用显存。CPU 功耗由 `cpu-power-limit.service` 限制（PL1 125 W，关睿频），不要改动。
- 用户没要求时不要启动完整流程；用户说先改完再跑时，不要中途重跑。

## 代码结构约定

- `graph.py` 只负责编排节点。节点返回 state 增量，数据落盘到任务目录的 `state/`，图片落盘到 `visualization/`，目录名和文件名统一在 `output_layout.py` 定义，路径一律相对于项目根目录。
- 保留原始像素网格。分辨率依次取自 GeoTIFF、Web Mercator，最后才是带告警的配置回退值。保持块 ID、坐标和版本 2 manifest 的字段契约；块按需裁剪。
- 会改变节点结果的修改，都要递增 `global_config.PIPELINE_VERSION` 并在旁边写注释。版本号写入 `state.json` 和 manifest，用来区分不同算法跑出的结果。
- 配置项新增在 `GlobalPreprocessConfig`，要同时补校验、`pipeline.yaml` 注释和 README。删除配置项时，`from_recorded` 会忽略旧任务里残留的字段。

## 感知模型

- 按模型隔离：`percection_tool/magroad/`、`segearth_ov3/`、`depth_anything/` 各自保存推理代码、第三方源码和权重。不要把适配器或权重混放到公共目录。公共层只有 interfaces、factory、dummy，factory 延迟导入所选模型。
- MaGRoad（`segmenter: magroad_scene`）使用官方推理代码和经过 SHA256 校验的权重，Node 1b 按场景选择权重。保持 `strict=True` 加载、float32 推理和官方滑窗平均融合。模型输出的图经 degree-2 链压缩后再布块，几何交叉不能自动当成路口；拓扑定稿后，路线沿 SegEarth-OV3 road 类的中线走。
- KITTI 使用 `input_geo_img/` 下的 `kitti_0930_0033.tif`、`kitti_1003_0027.tif`、`kitti_1003_0034.tif`，配置都是 `pipeline.yaml`。这些图的 EPSG:4326/affine 对应约 0.05 m/px，必须保留 `resolution_source=geotiff` 和原图像素网格。
- 旧 PathFinder 流程已删除，不要恢复。

## VLM

- 模型在根目录 `qwen3827b-exl3-dflash2/`：Qwen3.8-27B EXL3 加 DFlash2 草稿模型，以及 ExLlamaV3 源码。这是独立的 git 仓库，不修改其中任何内容。
- ExLlamaV3 编译后安装进 `.venv`，由 `agent/vlm/exl3_server.py` 作为独立进程提供 OpenAI 兼容服务。服务进程不继承 `PYTHONPATH`，否则模型项目里为旧 torch 编译的扩展会遮蔽已安装的引擎。
- 以下特性必须保持：只监听回环地址、首次推理时才启动、用完回收服务进程、llguidance 结构化输出、每个请求单独设置 thinking 预算。
- Node 6、7、9 正式重跑使用 `protect_stage_outputs` 暂存旧产物。重跑抛出异常时恢复旧文件、manifest 和 state；诊断运行使用独立目录。
- VLM 进度按请求结果的 `status` 统计成功和失败。token 用量及耗时用于速度统计。
- 原 Qwen3.5-9B + vLLM 后端已删除，不要恢复。
- prompt 系统放在 `agent/vlm/` 下：
  - `scene_type/`：Node 1b；
  - `block_review/semantic|height/`：Node 6；
  - `block_expansion/`：Node 7；
  - `localization_recommendation/`：Node 9。

  每套由 prompt.md（任务与输入含义）、skill.md（观察步骤）、rule.yaml（判据与程序映射）、schema.json（输出结构）组成。Node 7 复用 Node 6 的核心 rule 和判定字段。

## 退化分析分支（待实现）

- 计划为 `agent/run_agent.sh` 增加 `--analysis {auto,semantic,height,both}`。命令示例见根目录 [README](README.md#语义与高程分析模式拟新增)。
- `auto` 为默认值，沿用有 DSM 走高程、无 DSM 走语义的逻辑。`semantic` 只判语义；`height` 和 `both` 需要可用 DSM，支持自动配对及显式 `--dsm`。
- Node 5 对所选分支各自执行现有初筛规则。Node 6 仅为命中分支构造输入并独立请求 VLM。同一块同时命中时使用两套输入和 prompt。Node 7 只扩展 Node 6 确认退化的分支。

## VLM 输入图与介绍（Node 6 / Node 7）

- 模型只收到“图片 + 图片介绍”：
  - 介绍里写清每张图是什么、覆盖多大范围、颜色和透明度代表什么；
  - 统计值、参考高程、坐标等数据只写进 `<type>_metadata.json`，不发给模型；
  - 模型不知道 Node 5 的初筛结论。
- 输入图上不画道路拓扑线。Node 6 的局部图不加任何标记；Node 7 只画观察中心十字和档位框。
- 图片介绍由程序生成：`nodes/vlm_review.py` 中的 `model_input`、`color_description`、`DSM_DESCRIPTION`，以及 `nodes/vlm_expansion.py` 中的 `model_input`。prompt.md 的“输入”一节只写通用说明。
- 修改输入图时，下面几处要一起改，保持一致：
  - 图片张数和顺序；
  - `review_image_labels`；
  - 图片介绍；
  - prompt.md 和 skill.md 的输入说明；
  - schema 里 `observations` 的字段（每张图一个，顺序与图片相同）；
  - 测试；
  - README。
- 用户说“规则判定方式先不变”时，只改描述图片的措辞，不改 rule 的判定条件、schema 枚举和推导逻辑。rule.yaml 用物理含义描述高出、低处和无数据，具体颜色对应由图片介绍给出。
- 现行约定：
  - 尺寸：小 Patch 75 m（`block_size_m`），大 Patch 100 m（`vlm_expansion_sizes_m: [100]`）。
  - 语义颜色：道路为红色。
  - 语义上下文图：RGB 占 70%、语义色占 30%（`CONTEXT_TRANSPARENCY`）。
  - 高程着色：DSM 减去 100 m 范围内的 P20，按固定的 [-2, 8] m 线性映射到 Viridis，Node 6 和 Node 7 共用。

## 检查与文档

修改后运行离线回归，并用 `git diff --check` 检查格式：

```bash
.venv/bin/python -m unittest discover -s agent/map_preprocess/tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

离线测试用的是合成数据和假 VLM，不能证明真实模型的效果，汇报时要说明这一点。修改接口、配置或产物时，同步更新对应的 README（根目录、`agent/map_preprocess/`、`agent/vlm/`）。权重、虚拟环境、日志和运行产物按现有的 `.gitignore` 管理。

## 目录分工

- `qwen3827b-exl3-dflash2/`：不改动。
- `percection_tool/`：网络适配、第三方源码和权重。
- `agent/`：LangGraph、配置、state、VLM 客户端。
- `input_geo_img/`：输入，平铺 GeoTIFF。`<name>_dom.tif` 与 `<name>_dsm.tif` 配对；没有 `_dom` 后缀的图按没有 DSM 处理。
- `output/`：运行产物。

旧 manifest 中的路径通过 `agent.paths` 兼容，不改写历史记录；新的 Python 导入使用新包名。
