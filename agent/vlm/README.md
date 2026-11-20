# 本地 VLM 推理组件

本目录是本地 VLM 推理层；模型为根目录 `qwen3827b-exl3-dflash2/` 中的 Qwen3.8-27B EXL3 4.00bpw（视觉 6bpw）+ DFlash2 草稿模型投机解码，引擎为该项目锁定的 ExLlamaV3 community 分支，编译安装在项目 `.venv`（与感知、流程共用）。神经网络道路工具在 `../../percection_tool/`。原 Qwen3.5-9B + vLLM 后端已删除。

| 路径 | 职责 |
| --- | --- |
| `client.py` | 本地 HTTP 客户端、图片编码、结构化输出和请求限流 |
| `runtime.py` | 首次推理时用同一个 Python 启动 `exl3_server.py` 独立进程，日志和服务回收 |
| `exl3_server.py` | OpenAI 兼容服务（只实现客户端用到的接口）：多图交替消息、llguidance JSON Schema 约束、两段式 thinking 预算 |
| `common.py` | 业务 JSON、几何和配置；基础路径复用 agent.paths |
| `block_review/semantic/`、`block_review/height/` | Node 6 退化块复核的两套独立 prompt 系统（prompt/skill/rule/schema），由 `agent/map_preprocess/nodes/vlm_review.py` 调用 |
| `scene_type/` | Node 1b 整图道路结构（结构化/非结构化）判断的 prompt 系统（prompt/skill/schema），程序 `decide_scene` 映射到 MaGRoad 权重，由 `agent/map_preprocess/nodes/scene_type.py` 调用 |
| `block_expansion/` | Node 7 退化区域尺寸扩展的 prompt 系统：按支路的 prompt/skill，共用的多尺寸 rule/schema；核心判据和字段复用 `block_review`，多尺寸 Rule 另提供当前支路的环带与整窗判据，由 `agent/map_preprocess/nodes/vlm_expansion.py` 调用 |
| `localization_recommendation/` | Node 9 从俯视 RGB 推测车端几何约束并推荐辅助定位方式的 prompt 系统（prompt/skill/rule/schema），由 `agent/map_preprocess/nodes/localization_recommendation.py` 调用 |
| `parallel_vlm.py`、`loc_type_sys/` | 后续定位评估接口，配置未补齐，当前预处理不调用 |
| `.env.example` | 环境变量示例，不自动加载 |

### 环境

Agent、感知工具和 VLM 服务共用项目根目录 `.venv`（Python 3.11、torch 2.10.0+cu128、Triton 3.6）。官方环境为 torch 2.10 + CUDA 13，需要驱动 ≥ 580；本机驱动 535（CUDA 12.2）通过 CUDA 12 小版本兼容运行 cu128。服务进程不继承 `PYTHONPATH`，因为模型项目 `exllamav3/` 源码目录里有为旧 torch 编译的扩展，放进路径会遮蔽已安装的引擎。

```bash
/home/ubuntu/anaconda3/envs/agent/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
# 引擎：从模型项目源码的干净副本编译（不带其 build/ 和旧 .so），只编 sm_86
rsync -a --exclude build --exclude '*.so' --exclude '*.egg-info' --exclude __pycache__ \
    qwen3827b-exl3-dflash2/exllamav3/ /tmp/exllamav3-src/
CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=16 \
    .venv/bin/python -m pip install --no-build-isolation --no-deps /tmp/exllamav3-src
```

模型项目的 `exllamav3/` 更新后需要按上面重新编译安装。

实测（2026-11-11，官方 `scripts/acceptance_check.py`，GSM8K 40 题、贪心、512 新 token）：接受长度 5.658（官方 5.657）；关闭桌面程序时 DFlash2 149.5 tok/s、无草稿 41.4 tok/s（官方 162.9 / 42.8）。本机 3090 同时驱动桌面，gnome-shell、VS Code 等图形进程在 GPU 上抢时间片，桌面活跃时约降到 127 / 34.6 tok/s；改用核显出图可消除。

### EXL3 服务

`exl3_server.py` 不导入 `agent` 包，由 `runtime.py` 以脚本路径在独立进程中启动（与 Agent 同一个 Python，进程结束即释放显存），只监听 127.0.0.1。一个生成线程独占 GPU：视觉编码、模板渲染、入队和生成都在该线程，HTTP 线程只做解析和图片解码。

- 消息：`image_url` 只接受 `data:image/...;base64`；按原顺序换成模型模板的图片占位，再整段替换为 ExLlamaV3 图片嵌入（嵌入自带 vision_start/end）。
- 结构化输出：`response_format` 的 `json_schema` / `json_object` 编译为 llguidance 语法，编译失败返回 400；约束只作用于回答，不作用于 thinking。
- thinking：`chat_template_kwargs.enable_thinking`（未指定时按模板默认开启）。开启时分两段生成：第一段遇 `</think>` 或达到 `thinking_token_budget` 停止；预算用尽时补 `\n</think>\n\n` 强制闭合，第二段在其后生成回答。推理文本在 `message.reasoning_content`；`</think>` 计入 completion token。thinking 未闭合就用尽 `max_tokens` 时 `finish_reason=length`。开启 thinking 时模型模板默认在 system 前加 `reasoning_effort=xhigh` 说明，可经 `chat_template_kwargs.reasoning_effort`（xhigh/medium/low）调整。正式流程的各 VLM 节点都传 medium，模板采用普通思考提示，不额外加 xhigh 或 low 的提示。
- 响应额外带 `exl3` 字段（各阶段 prefill/生成耗时、命中缓存和草稿接受 token 数），便于调参。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `VLM_MODEL_ROOT` | `qwen3827b-exl3-dflash2` | 模型项目根目录（含 `models/`） |
| `VLM_PYTHON` | 当前 Python（`sys.executable`） | 服务进程解释器；默认与 Agent 同一环境 |
| `VLM_MODEL_PATH` / `VLM_DRAFT_PATH` | `<root>/models/qwen38-27b-exl3` / 并行关闭，串行使用配套 DFlash2 | 目标/草稿模型；`VLM_DRAFT_PATH=none` 显式关闭投机解码 |
| `VLM_CACHE_TOKENS` | 五路默认 73728；一至三路默认 32768 | KV cache 总 token（各请求共享显存空间，模型输入和输出各自独立）；缓存不足时请求排队 |
| `VLM_CACHE_BITS` | 3 | KV cache 量化位数，0 = fp16 |
| `VLM_MAX_NUM_SEQS` | 5 | 服务端同时生成的序列数，也是客户端并发上限；实际并发还取决于节点的 worker 数和可用缓存 |
| `VLM_MAX_MODEL_LEN` | 32768 | 单请求 prompt + max_tokens 上限，不得超过 cache |
| `VLM_MAX_IMAGES` / `VLM_MAX_IMAGE_PIXELS` | 4 / 1048576 | 客户端和服务端同用的单请求图片数和单图像素上限 |
| `VLM_START_TIMEOUT` / `VLM_READY_TIMEOUT` / `VLM_TIMEOUT` | 900 / 60 / 1800 秒 | 启动等待、繁忙探活容忍、单请求 HTTP 超时 |

`EXL3_HGEMM_F16ACC=auto`、`EXL3_INT8_GEMV=2`、`EXL3_INT8_GEMV_MAX_K=5`、`EXL3_QC_STAGING=1` 为 3090 默认值，外部已设置时保留外部值。以上并发、cache 和上下文默认值为迁移占位，尚未按本流程实测调参。

调用入口：`from agent.vlm.client import VLMClient`，使用 `with VLMClient() as client:` 管理自己启动的服务。多图请求可传 `image_labels=[...]`（与 `images` 等长的非空字符串）：每张图前插入对应文字标签，user 消息变为“标签1、图1、标签2、图2 …、主提示词”；不传时仍为“全部图片 + 主提示词”。`enable_thinking=True, thinking_token_budget=N` 开启受限 thinking，推理文本在 `result.reasoning`（需 `VLM_AUTOSTART=1`，由本程序启动的 exl3_server 保证预算生效）。HTTP 仅使用本机回环地址，首次实际推理才启动；服务启动后探活（`/models`）在繁忙时可能变慢，进程仍在时最多等待 `VLM_READY_TIMEOUT` 秒（默认 60）才判为不可用。Agent 依赖仍为根目录 `requirements.txt` 与 `.venv`；服务进程依赖由模型项目 `.venv` 提供（ExLlamaV3、llguidance、transformers、Pillow）。日志在 `output/vlm/log/`，历史记录在 `output/vlm/runtime/`。

迁移已通过离线回归（请求解析、thinking 分段、启动命令）以及用真实 tokenizer 的 CPU 检查（所有业务 Schema 可编译为 llguidance 语法、多图模板占位、`</think>` 单 token）；这不等价于加载完整模型进行真实推理。

Node 9 schema v3 分别记录 LIO 和轮速适用性，以及 GPS 的天空开阔程度和遮挡或反射线索。推荐按 LIO、轮速、GPS 的顺序校验。LIO 根据沿路的稳定几何选型，树荫和遮挡不参与评分。推荐理由须对应本区域已记录的图像证据。

`block_review` 的语义 schema 为 v11，高程为 v12。两条支路都返回图像证据、逐字段置信度及 `verdict`。程序校验数据可用性和证据引用后采用 `verdict.result` 作为退化结论。历史上没有 `verdict` 的结果仍可按旧规则解释。
语义关注类别占比、道路边界、类别侵入和直道沿路语义重复；高程关注高出结构分布及直道两侧持续相似的高度分布。
两套新增 `road_geometry`、`longitudinal_structure` 中文枚举，派生 `degradation_type` 区分退化、纵向退化、不退化与未知；不判断雷达实际可见性或定位切换。
未知数据不归为正常；建筑占比不再自动豁免。详见 [Node 6 说明](../map_preprocess/README.md#node-6退化块-vlm-复核vlm_review_blocks)。

`matched_reasons` 仅记录最终确认退化的原因；`candidate_reasons` 保留未确认候选，`consistency_failures` 记录局部矛盾。场景类型从支持通行结论的有效地面类型中选取。

Node 6 输入先统一裁去地图外区域，允许长方形；范围、横纵比例尺和中心位置随实际图像发送。地图内部原始无数据仍保留，具体字段见 Node 6 输入说明。

正式流程的 75 m 基准框来自 Node 4 沿道路拓扑按 48.75 m 弧长步长布出的框。Node 5 按阈值筛选候选块，Node 6 独立复核候选块的对应支路。

Node 6 的 `<type>_input.json` 仅包含标识及图像说明（semantic、height 各三图），几何与统计留在 `<type>_metadata.json`。请求正文只给实际覆盖宽高及颜色解释，完整 Schema 通过结构化输出参数传入。

Node 6 高程复核接收 Node 5 阈值命中的块及道路高程基准无效但仍有有效 DSM 的块。两种情况使用相同的图像、着色参考和规则。道路基准检查及初筛结果保存在 Node 5 产物中，模型根据输入图独立判断。

道路中心相对有效路面基准出现向上阶跃时，Node 6 高程请求附上遮挡、车辆可进入的隧道或下穿入口、建筑占路的核查提示。

`--analysis semantic|height|both` 控制 Node 5 的初筛分支。Node 6 按候选支路分别构造输入，Node 7 只扩展已确认退化的支路。
