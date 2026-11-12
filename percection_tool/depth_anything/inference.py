"""官方 Transformers 版 V2 Large 本地离线推理；输出无米制单位的相对逆深度。"""
from __future__ import annotations
import math
import time
from typing import TYPE_CHECKING
import numpy as np
from agent.paths import resolve_path
from percection_tool.interfaces import DepthResult

if TYPE_CHECKING:
    from agent.perception_config import DepthAnythingConfig


class DepthAnythingEstimator:
    def __init__(self, config: DepthAnythingConfig | None = None):
        if config is None:
            from agent.perception_config import DepthAnythingConfig
            config = DepthAnythingConfig()
        self.settings = config
        self.model = self.processor = self.device = None
        self.load_report = None

    def _load(self) -> None:
        from .download import FILES, REVISION, REPOSITORY, verify
        directory = resolve_path(self.settings.weights_dir)
        for name,expected in FILES.items():
            if not verify(directory/name,expected):
                raise ValueError(f'Depth Anything 本地文件缺失或校验失败：{directory/name}；运行 python -m percection_tool.depth_anything.download')
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        requested = self.settings.device
        if requested == 'auto':
            requested = ('cuda:0' if torch.cuda.is_available() and
                torch.cuda.mem_get_info(0)[0] >= self.settings.min_free_memory_mib*1024**2 else 'cpu')
        device = torch.device(requested)
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('指定 CUDA 不可用')
            if torch.cuda.mem_get_info(device)[0] < self.settings.min_free_memory_mib*1024**2:
                raise RuntimeError('Depth Anything 可用显存低于配置下限')
        processor = AutoImageProcessor.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False,use_fast=False)
        model,report = AutoModelForDepthEstimation.from_pretrained(str(directory),local_files_only=True,
            trust_remote_code=False,use_safetensors=True,output_loading_info=True)
        if any(report.get(name) for name in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError(f'Depth Anything 权重未完整匹配：{report}')
        self.model, self.processor, self.device = model.to(device).eval(), processor, device
        self.load_report = {'repository':REPOSITORY,'revision':REVISION,'dtype':'float32',**report}

    def predict(self, image: np.ndarray, valid_mask: np.ndarray | None = None) -> DepthResult:
        if not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError('深度输入必须为非空 HWC uint8 RGB')
        height,width = image.shape[:2]
        if height*width > self.settings.max_input_pixels:
            raise ValueError('深度输入超过 max_input_pixels；请按需裁剪块')
        if valid_mask is not None and (valid_mask.shape != (height,width) or valid_mask.dtype != np.bool_):
            raise ValueError('valid_mask 必须为同尺寸 bool 数组')
        # 先估计保持纵横比、14 对齐后的处理网格，避免超长图在预处理时分配巨量张量。
        scale = max(self.settings.input_size/height,self.settings.input_size/width)
        estimated = math.ceil(height*scale/14)*14*math.ceil(width*scale/14)*14
        if estimated > self.settings.max_processed_pixels:
            raise ValueError('Depth Anything 处理网格超过 max_processed_pixels')
        started = time.monotonic()
        if self.model is None:
            self._load()
        import torch
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self.settings.threads)
        try:
            inputs = self.processor(images=image,return_tensors='pt',size={'height':self.settings.input_size,'width':self.settings.input_size})
            if inputs['pixel_values'].shape[-2]*inputs['pixel_values'].shape[-1] > self.settings.max_processed_pixels:
                raise ValueError('Depth Anything 实际处理网格超过上限')
            inputs = {key:value.to(self.device) for key,value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs).predicted_depth
                depth = torch.nn.functional.interpolate(output[:,None],size=(height,width),mode='bicubic',align_corners=False)[0,0].float().cpu().numpy()
        finally:
            torch.set_num_threads(previous_threads)
        if not np.isfinite(depth).all():
            raise RuntimeError('Depth Anything 输出非有限值')
        if valid_mask is not None:
            depth[~valid_mask] = np.nan
        return {'depth':depth,'metadata':{'model':'Depth Anything V2 Large','representation':'relative_inverse_depth',
            'units':'relative','larger_is':'nearer','is_metric':False,'is_elevation':False,
            'device':str(self.device),'runtime_torch_version':torch.__version__,'shape':[height,width],'processed_shape':list(inputs['pixel_values'].shape[-2:]),
            'input_size':self.settings.input_size,'invalid_pixels':int((~valid_mask).sum()) if valid_mask is not None else 0,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
            'elapsed_seconds':time.monotonic()-started,'loading':self.load_report}}

    def close(self) -> None:
        self.model = self.processor = self.device = None
