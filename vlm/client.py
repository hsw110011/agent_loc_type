"""无云端依赖的 vLLM HTTP 客户端；所有业务节点复用此入口。"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageOps
from .runtime import VLMRuntime


class VLMError(RuntimeError):
    """传输失败、模型输出截断或结构不符合约定。"""


@dataclass(frozen=True)
class VLMConfig:
    base_url: str = 'http://127.0.0.1:8011/v1'
    model: str = 'local-qwen'
    timeout: float = 180.0
    max_tokens: int = 1024
    max_images: int = 4
    max_image_pixels: int = 1048576
    api_key: str = field(default='', repr=False)

    def __post_init__(self):
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('仅允许本机 HTTP 服务，避免图片被发送至外部地址')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('base_url 不得包含凭据、查询参数或片段')
        if self.timeout <= 0 or self.max_tokens <= 0 or self.max_images <= 0 or self.max_image_pixels <= 0:
            raise ValueError('超时、token 数、图片数与像素限制必须为正数')

    @classmethod
    def from_env(cls) -> 'VLMConfig':
        """只读取进程环境；不读取已有项目 .env，也不读取 OPENAI_API_KEY。"""
        return cls(
            base_url=os.getenv('VLM_BASE_URL', 'http://127.0.0.1:8011/v1'),
            model=os.getenv('VLM_MODEL', 'local-qwen'),
            timeout=float(os.getenv('VLM_TIMEOUT', '180')),
            max_tokens=int(os.getenv('VLM_MAX_TOKENS', '1024')),
            max_images=int(os.getenv('VLM_MAX_IMAGES', '4')),
            max_image_pixels=int(os.getenv('VLM_MAX_IMAGE_PIXELS', '1048576')),
            api_key=os.getenv('VLM_API_KEY', ''),
        )


@dataclass(frozen=True)
class InferenceResult:
    text: str
    usage: dict[str, Any]
    latency_seconds: float
    request_id: str
    finish_reason: str


class VLMClient:
    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig.from_env()
        self.runtime = VLMRuntime(self.config.base_url, self.config.model)
        concurrency = int(os.getenv('VLM_MAX_NUM_SEQS', '2'))
        if concurrency <= 0:
            raise ValueError('VLM_MAX_NUM_SEQS 必须为正整数')
        self._slots = threading.BoundedSemaphore(concurrency)
        # 明确绕过代理环境，且禁止重定向将本地图片发送到其它服务。
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def close(self):
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _request(self, path: str, payload: dict | None = None) -> dict:
        headers = {'Content-Type': 'application/json'}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(self.config.base_url.rstrip('/') + path, data=body, headers=headers)
        try:
            with self._http.open(req, timeout=self.config.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode('utf-8', errors='replace')
            raise VLMError(f'vLLM HTTP {exc.code}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise VLMError(f'本机 vLLM 请求失败: {exc}') from exc

    def health(self) -> dict:
        result = self._request('/models')
        if self.config.model not in [item['id'] for item in result.get('data', [])]:
            raise VLMError(f'服务未提供模型 {self.config.model}')
        return result

    def _prepare_image(self, path: str | Path) -> str:
        from .common import resolve_path
        with Image.open(resolve_path(path)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        if image.width * image.height > self.config.max_image_pixels:
            ratio = math.sqrt(self.config.max_image_pixels / (image.width * image.height))
            image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                                 Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')

    def _infer(self, prompt: str, images: Sequence[str], *, system: str,
               max_tokens: int | None, temperature: float, json_output: bool,
               output_schema: dict | None = None, request_path: Path | None = None,
               response_path: Path | None = None) -> InferenceResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt 不能为空')
        if len(images) > self.config.max_images:
            raise ValueError(f'每次最多允许 {self.config.max_images} 张图片')
        tokens = self.config.max_tokens if max_tokens is None else max_tokens
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError('max_tokens 必须为正整数')
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature 必须为非负有限数值')
        content = [{'type': 'image_url', 'image_url': {'url': image}} for image in images]
        content.append({'type': 'text', 'text': prompt})
        payload = {
            'model': self.config.model,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            'max_tokens': tokens, 'temperature': temperature, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        if output_schema is not None:
            if not isinstance(output_schema, dict):
                raise ValueError('output_schema 必须为 JSON Schema 对象')
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': {'name': 'vlm_output', 'schema': output_schema, 'strict': True},
            }
        elif json_output:
            payload['response_format'] = {'type': 'json_object'}
        if request_path is not None:
            from .common import write_json
            write_json(request_path, payload)
        started = time.monotonic()
        with self._slots:
            try:
                self.runtime.ensure_started()
            except (OSError, ValueError, RuntimeError) as exc:
                raise VLMError(str(exc)) from exc
            data = self._request('/chat/completions', payload)
        if response_path is not None:
            from .common import write_json
            write_json(response_path, data)
        elapsed = time.monotonic() - started
        try:
            choice = data['choices'][0]
            text = choice['message']['content']
            reason = choice['finish_reason']
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError('vLLM 返回了不完整的响应结构') from exc
        if reason != 'stop':
            raise VLMError(f'推理未正常完成，finish_reason={reason}；可检查输出 token 上限')
        if not isinstance(text, str) or not text.strip():
            raise VLMError('模型返回空文本')
        return InferenceResult(text, data.get('usage', {}), elapsed, data.get('id', ''), reason)

    def infer(self, prompt: str, images: Sequence[str | Path] = (), *,
              system: str = '你是视觉分析助手。根据图片与任务给出准确回答，不猜测不可见内容。',
              max_tokens: int | None = None, temperature: float = 0.0,
              json_output: bool = False, output_schema: dict | None = None) -> InferenceResult:
        """通用节点接口；支持纯文本、单图、多图。不会自动重试昂贵的推理请求。"""
        if isinstance(images, (str, Path)):
            raise ValueError('images 请使用列表，例如 [image_path]')
        if len(images) > self.config.max_images:
            raise ValueError(f'每次最多允许 {self.config.max_images} 张图片')
        prepared = [self._prepare_image(path) for path in images]
        return self._infer(prompt, prepared, system=system, max_tokens=max_tokens,
                           temperature=temperature, json_output=json_output, output_schema=output_schema)

