# SegEarth-OV3 语义分割

流程（Node 2b、5、6、7）默认使用的语义分割工具，只输出道路、建筑、植物三类。滑窗融合移植自
`SegEarth-OV-3/segment_geotiff.py` 和 `segearthov3_segmentor.py`，不依赖 mmseg，运行时也不访问源工程。
与该脚本的差别：不查询 background、water，也不按分数阈值判背景，每个有效像元都归入分数最高的一类。

- `vendor/sam3/`：上游 `sam3` 包原样复制，版本和文件哈希见 `weights/SOURCE.json`。
- `weights/sam3.pt`：约 3.3 GB，由下面的命令导入，每次加载校验 SHA256，不进 git。
- `inference.py`：`SegEarthOV3Segmenter`，实现 `SemanticPredictor` 协议。

```bash
.venv/bin/python -m percection_tool.segearth_ov3.import_model [--source-dir ../Geoseg_hsw/SegEarth-OV-3]
```

## 推理

窗口 1600、步长 1200，重叠带线性加权；不足窗口的输入在右下补零。7 行提示词共 11 个查询词，同行取最大分数后再比较，
bfloat16 autocast，仅支持 CUDA。配置在 `agent/perception_tools.yaml` 的 `segearth_ov3` 段，
`pipeline_semantic: user_unetformer` 可切回用户三类模型。

| 提示词 | 输出 |
| --- | --- |
| road、car | 0 road |
| building,roof,house | 1 building |
| bareland,barren、grass、tree,forest、cropland | 2 plant |
| `valid_mask` 为假或纯黑像元 | 255 无效 |

`confidence` 是最高分到 [0.5, 1] 的线性映射，不超过 `score_threshold`（0.1）的分数取 0.5。流程的
`feature_min_class_confidence` 默认 0.5，因此不会把像元降为 other。`score_threshold` 同时是 SAM 3 实例筛选的置信度阈值。

## 资源

运行上游脚本时观察到进程占用约 16 GB 显存，`min_free_memory_mib` 默认 16384，不能与 VLM 或其它大模型同时占用 GPU。
上游仓库未附 LICENSE 文件，SAM 3 权重的使用条款以 Meta 发布为准。
