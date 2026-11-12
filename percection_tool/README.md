# 神经网络感知工具

道路拓扑模型 MaGRoad与后续分析模型 Depth Anything V2 Large、语义分割 SegEarth-OV3分别独立存放。道路配置由 Agent 的 `PreprocessConfig` 传入，分析工具配置见 `agent/perception_tools.yaml`；均复用根目录依赖与现有环境。

```text
percection_tool/
├── interfaces.py       # 输入/输出协议
├── factory.py          # 仅延迟选择所需工具，不包含模型实现
└── dummy.py            # 仅供合成图测试
```

直接使用：`from percection_tool.magroad.inference import MaGRoadExtractor`（整图流程经 `factory.create_graph_extractor` 调用）。各工具不相互导入，不共用权重目录。导入及创建工具不会加载权重或启动 VLM；模型在实际预测时延迟加载。


此前 manifest/config 内的旧权重路径由 `agent.paths.resolve_path` 映射到各模型的新目录；Python 导入使用以上新包名。LangGraph 节点与布块处理位于 `agent/map_preprocess/`。

## MaGRoad 多权重道路图（流程默认）

`magroad/`：MaGRoad 官方 WildRoad / Global-Scale / SpaceNet 三个权重的纯 PyTorch 推理，按各自训练 GSD
重采样后提取道路图。可单独运行对比，也可经 `graph_adapter.py` 接入整图流程（`segmenter: magroad_scene`，由 Node 1b VLM 场景判断选权重，factory 延迟导入）。详见 `magroad/README.md`。

```bash
.venv/bin/python -m percection_tool.magroad.download
.venv/bin/python -m agent.run_magroad --image input_geo_img/miluo_dom.tif
```

## 后续分析用独立工具

- `depth_anything/`：官方 Depth Anything **V2 Large**，本地权重、下载校验、相对逆深度推理。
- `segearth_ov3/`：SegEarth-OV3（SAM 3 开放词汇）语义分割，流程默认使用；由 `agent/perception_tools.yaml` 的 `pipeline_semantic` 切换。

两者均延迟加载，factory 只导入所选模型；不会触发 VLM、训练或道路预处理。
配置集中在 `agent/perception_tools.yaml`，默认值在 `agent/perception_config.py`。
统一独立入口：

```bash
# 两种模型顺序运行
.venv/bin/python -m agent.run_perception --tool both --image input_geo_img/example.png

# 后续分析直接读取 manifest 的一个块，包括显式边界/无效像元掩码
.venv/bin/python -m agent.run_perception --tool both \
  --manifest output/map_preprocess/global_blocks/kitti_0930_0033/<task_id>/state/manifest.json \
  --block-id E30_B0 --config agent/perception_tools.yaml
```

输出独立保存在 `output/perception_tools/<图片名或图片名_块ID>/<任务ID>/`：
`input.png,valid_mask.png,config.json,result.json`，以及所选工具的 `depth.npy,
depth_preview.png,semantic_mask.png,semantic_color.png,semantic_overlay.png,
semantic_confidence.npy`。模型失败保留 error.json 和已完成工具的产物。
`result.json` 存文件路径和轻量信息，不存大数组。输入超过工具 max_input_pixels
会报错，请按需裁剪，不默认把整幅大图缩小或批量处理所有块。

整幅 GeoTIFF 直接生成相对深度 GeoTIFF（只跑深度，不切块）：

```bash
.venv/bin/python -m agent.run_depth_geotiff   # 默认 input_geo_img/kitti_0930_0033.tif
.venv/bin/python -m agent.run_depth_geotiff --image <图> --output <输出.tif> --input-size 3360
```

默认写到 `output/perception_tools/whole_image/<图片名>_depth.tif`，同名文件会被覆盖。输出为单波段
float32，CRS、仿射变换和像素网格与输入一致，无效像元为 NaN；输入必须带地理参考。整图只做一次
前向，细节上限由 `--input-size`（默认 3780）决定，原网格上的值是插值结果。
KITTI 这张图在 RTX 3090、FP32 下实测：

| input_size | 处理网格 | 约合 m/px | predict 耗时 | CUDA 峰值 allocated | 可见细节 |
| --- | --- | --- | --- | --- | --- |
| 518 | 518×560 | 1.1 | 9.5 s | 1.8 GiB | 建筑成团、路网可辨 |
| 1890 | 1890×2030 | 0.3 | 15 s | 5.6 GiB | 屋顶轮廓、单棵树冠 |
| 2800 | 2800×3010 | 0.2 | 38 s | 10.8 GiB | 屋面结构 |
| 3360 | 3360×3612 | 0.17 | 65 s | 15.0 GiB | 路边车辆、树冠纹理 |
| 3780 | 3780×4074 | 0.15 | 99 s | 18.7 GiB | 与 3360 接近，收益已小 |

以上为目视对比，没有深度真值，不代表精度。5040 未在 170 s 内跑完，未验证。

整图流程 Node 5（`block_features_enabled: true`）逐块调用这两个工具；`run_perception` 仍可单独对图片或块运行。

### 本次部署验证

现有环境 torch 2.10.0+cu128、Transformers 4.57.6，未安装或升级任何包。
56 项道路预处理测试与 19 项公共/工具测试通过，新增工具 7 项测试包含 RGB 字节
输入、滑窗融合/填充、无效像元、非有限输出、线程恢复、延迟隔离加载、文件完整性
及分段断点续传。`git diff --check` 和新增文件空白检查通过。

在真实 KITTI GeoTIFF 的 manifest 块 E30_B0（100m、2000×2000 原网格）上，使用完整
实际权重执行两个模型：Depth Anything V2 Large FP32 输出有限的 2000×2000 相对
逆深度，严格加载无缺失/多余/不匹配权重；用户语义模型完成 49 个滑窗，输出同网格
三类标签。单进程顺序调用，predict 含模型加载耗时分别约 5.57s / 10.35s，推理阶段
峰值 CUDA allocated 分别约 1.57 / 1.76 GiB；这些值仅对应本次尺寸和设备。
原用户部署的一个既有 C++ 验证样本与当前语义适配器逐像素类别一致率为 100%，属于
运行时一致性检查，不代表新的分割精度。没有深度真值，不填写深度/高程精度。

运行结果：
`output/perception_tools/real_smoke/blocks/kitti_raw_0930_0027_E30_B0/4922bdc611a24e558569acfe5eaa62df/result.json`。
总体验证记录：`output/perception_tools/real_smoke/verification.json`；额外 640×576
原图裁剪验证了四窗口语义拼接，来源坐标保存在同目录 source.json。
