# MaGRoad 道路拓扑（多权重）

复现 [MaGRoad](https://github.com/xiaofei-guan/MaGRoad)（*Beyond Endpoints: Path-Centric Reasoning for
Vectorized Off-Road Network Extraction*，arXiv 2512.10416）的推理，使用 README Model Zoo 公布的三个官方权重。
与 `sam_road/` 相互独立，不互相导入。

```text
magroad/
├── inference.py   # MaGRoadExtractor(checkpoint).predict_graph(rgb, gsd, valid_mask)
├── download.py    # Google Drive 下载、续传、大小 + SHA256 校验
├── graph_adapter.py  # 整图流程 Node 2 适配（segmenter: magroad_scene），输出统一道路图
├── vendor/        # 上游 model.py 原文、AST 提取的推理模型、SAM ViT 编码器、官方配置、许可证、SOURCE.json
└── weights/       # 三个 checkpoint 及 *.source.json（git 忽略）
```

| 名称 | 训练数据 | 训练 GSD | patch | 来源 |
| --- | --- | --- | --- | --- |
| `wildroad` | WildRoad 越野道路 | 0.3 m/px | 1024 | main 分支 |
| `globalscale` | Global-Scale 城乡道路 | 1.0 m/px | 512 | main 分支 |
| `spacenet` | SpaceNet 城市道路 | 1.0 m/px | 256 | cityspace 分支 |

```bash
.venv/bin/python -m percection_tool.magroad.download            # 全部，或 --checkpoint wildroad
.venv/bin/python -m agent.run_magroad                           # 默认 data/汨罗遥感/miluo_dom_0.05m.tif，三个权重
.venv/bin/python -m agent.run_magroad --checkpoints wildroad --image <GeoTIFF>
# 下调阈值（只影响取点/连边后处理）；同一次运行可并列官方与下调变体，输出目录名带阈值后缀
.venv/bin/python -m agent.run_magroad --checkpoints globalscale globalscale:ROAD=0.1,ITSC=0.05,TOPO=0.3
```

阈值键为 `ITSC`（路口点）、`ROAD`（道路点）、`TOPO`（连边），取值 (0, 1)；官方值与实际值都写入
`graph.json`/`summary.json`（`official_thresholds`/`thresholds`）。官方值是各数据集验证集上 F1 最优点，
跨域影像（如 0.05 m 无人机正射图）上 Global-Scale 的道路概率整体偏低，下调可提高召回但会增加假边。

## 接入整图预处理

`segmenter: magroad_scene`（`agent/map_preprocess/pipeline.yaml`）：Node 1b 由本地 VLM 判断整图
结构化/非结构化，非结构化用 `wildroad`（官方阈值），结构化用 `globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455`；Node 2 通过
`MaGRoadGraphExtractor` 输出与 SAM-Road 相同格式的 `raw_road_graph.json`，概率图存于道路图缓存目录
（`road_probability.npy/png`、`keypoint_probability.npy/png`、`magroad_native.json`）。详见 `agent/map_preprocess/README.md`。

## 与上游 inferencer.py 的对应

- 网络：`vendor/build_inference_model.py` 从上游 `model.py` 机械提取 `MaGRoad.__init__`（截至损失定义之前）、
  `infer_masks_and_img_features`、`infer_toponet` 及其依赖类，去掉 Lightning/torchmetrics/detectron2。
  完整 checkpoint 以 `strict=True` 加载，float32 推理。WildRoad 权重额外带训练损失 buffer
  `mask_criterion.pos_weight`（`BCE_POS_WEIGHT: 10`），加载前仅剔除 `mask_criterion/mask_dice_criterion/topo_criterion`
  前缀的键并记入 `checkpoint_load.dropped_training_only_keys`，网络参数仍全部 strict 匹配。
- 配置：`vendor/configs/` 与上游逐字节一致，读取语义复现 addict（缺省键为假值）。仅 `spacenet` 有覆盖，
  依据是权重的实际结构，否则 strict 加载失败：
  - `USE_GEOMETRIC_FEATURES: True`：该权重由 cityspace 分支训练，其 MaGTopoNet 恒含 `geo_proj`；
  - `USE_POINT_FEATURES: True`：公开权重含 `topo_net.node_proj`，`edge_proj` 输入 768 = 2×256+128+128，
    与公开配置写的 False 不符。
  cityspace 配置无 `MODEL_NAME`，按其 inferencer 视为 MaGRoad。
- 后处理：概率融合 → ×255 截断为 uint8 → `use_fast_nms=False` 的三遍 NMS → patch 内 kNN
  （`MAX_NEIGHBOR_QUERIES`、`NEIGHBOR_RADIUS`）→ MaGTopoNet（用该 patch 的 mask_logits 计算路径特征）→
  按有向对平均分数并以 `TOPO_THRESHOLD` 过滤；导出无向边时取两个方向的并集及最大分数。

大图适配（上游脚本只针对各自数据集固定尺寸）：

- 输入先用 Lanczos 重采样到权重的训练 GSD，节点映射回原图像素边缘坐标 `(x+0.5)*scale`。
  在 0.05 m/px 原图上直接推理，道路宽度是训练分布的 6–20 倍，不作为默认。可用 `--model-resolution` 覆盖。
- 滑窗步长取上游在其测试图上的步长：`(参考短边 - PATCH)/(INFER_PATCHES_PER_EDGE-1)`，即 WildRoad 204.8、
  Global-Scale 102.4、SpaceNet 9.6 px；上游固定 16 个/边在大图上会留空洞。
- GeoTIFF 无效区（`dataset_mask`）的概率置零后再取点；patch 特征缓存在 CPU。若 `dataset_mask` 全有效
  （无 nodata/alpha，如河西 DOM），入口把开运算后的大片纯黑 (0,0,0) 区域视为无效，`summary.json` 记
  `valid_source=black_border_fallback`；`--keep-black` 可关闭。

## 输出

`output/perception_tools/magroad/<图片名>/<时间戳>/`：

- `comparison.jpg`：各权重道路图并排对比；`summary.json`：分辨率、各权重节点/边数、总长度、耗时、显存。
- `<权重>/graph.json`：节点（原图像素坐标与经纬度）、无向边及分数、有向预测、元数据（含 strict 加载报告）。
- `<权重>/roads.geojson`：EPSG:4326 线段；`road_probability.tif`/`keypoint_probability.tif`：工作网格上的
  概率×255，带缩放后的仿射变换；`graph_preview.png`、`graph_working_grid.png`、`road_probability.png`。

权重 SHA256 由首次下载计算后固定在 `inference.CHECKPOINTS`（上游未公布哈希）。
