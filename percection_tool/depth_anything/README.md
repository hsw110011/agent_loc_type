# Depth Anything V2 Large

采用官方组织的 Transformers 版本：
[Depth-Anything-V2-Large-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)，
DINOv2 ViT-L/14 + DPT，约 335M 参数。复用现有 Transformers 4.57.6、torch 2.10，
未安装/升级依赖；没有改写网络，也没有部署额外的远程服务。

固定 HF revision `7581137eff8d4e94f6e796d3baea0e9fa79b22d2`。
`download.py` 对模型验证完整 SHA256，对配置/处理器/模型卡验证官方 Git blob ID；
下载后保存各文件 SHA256 来源记录。大文件分段、断点续传，最终校验整个文件。
推理使用 `local_files_only=True`、`trust_remote_code=False`、safetensors，并拒绝
missing/unexpected/mismatched 权重，不会偷偷下载或随机初始化缺失层。

```bash
.venv/bin/python -m percection_tool.depth_anything.download
.venv/bin/python -m agent.run_perception --tool depth \
  --image output/perception_tools/real_smoke/kitti_rgb.png
```

```python
from agent.perception_config import DepthAnythingConfig
from percection_tool.factory import create_depth_estimator

tool = create_depth_estimator(DepthAnythingConfig(device='cuda:0', input_size=518))
try:
    result = tool.predict(rgb_uint8, valid_mask=valid_bool_mask)
    depth = result['depth']  # H×W float32，与输入 RGB 同网格
finally:
    tool.close()
```

输入 HWC uint8 RGB；官方处理器保持纵横比、按 14 对齐，默认目标 518。
`input_size` 可设 700 等 14 的倍数，但须留足计算资源；原图保留原始像素尺寸，
预测通过官方示例的 bicubic/align_corners=False 插值回原尺寸。
使用 FP32 推理，auto 根据显存选择 CUDA/CPU；指定 CUDA 不可用或真实 OOM 时明确失败，
不偷偷降低尺寸或精度。模型延迟加载，close 释放模型引用；CUDA 缓存仍由 PyTorch 管理。

**输出含义是相对逆深度，数值越大通常越近，无米制单位。** 不能直接当作 DSM、建筑
高度、地面高程或真实距离，跨块的原始尺度也不保证一致。顶视遥感效果需结合具体
场景验证，本次只接入通用预训练模型，没有遥感深度真值评测。
可选 bool 有效掩码将无效位置置为 NaN；预览为单次结果 min/max 拉伸的灰度图，
白色更近，精确原值保存在 `depth.npy`。metadata 明确保存 `is_metric=false`、
`is_elevation=false`、版本、设备、实际处理尺寸和推理期间 CUDA 峰值 allocated 字节。
`max_input_pixels` 限制原图规模，`max_processed_pixels` 通过保守估计及实际张量尺寸
双重保护处理网格；超出时请按需读取道路块。

权重授权为官方标注的 **CC-BY-NC-4.0**；来源、模型卡随本地权重保存。
实现由本环境安装的 Transformers 提供，权重位于本目录 `weights/`，不与道路模型混放。
