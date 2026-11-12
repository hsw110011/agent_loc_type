"""无云端依赖的本机 VLM HTTP 客户端（EXL3 服务，见 exl3_server.py）；所有业务节点复用此入口。"""
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


def _check_image_labels(labels: Sequence[str] | None, count: int) -> list[str] | None:
    if labels is None:
        return None
    if isinstance(labels, str):
        raise ValueError('image_labels 请使用列表，与 images 一一对应')
    labels = list(labels)
    if len(labels) != count:
        raise ValueError(f'image_labels 数量（{len(labels)}）必须与图片数量（{count}）一致')
    if any(not isinstance(v, str) or not v.strip() for v in labels):
        raise ValueError('image_labels 每项必须为非空字符串')
    return labels


class VLMError(RuntimeError):
    """传输失败、模型输出截断或结构不符合约定；已生成的文本保留在 raw_text / reasoning，便于排查。"""

    def __init__(self, message: str, *, raw_text: str = '', reasoning: str = ''):
        super().__init__(message)
        self.raw_text = raw_text
        self.reasoning = reasoning


@dataclass(frozen=True)
class VLMConfig:
    base_url: str = 'http://127.0.0.1:8011/v1'
    model: str = 'local-qwen'
    timeout: float = 1800.0
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
            timeout=float(os.getenv('VLM_TIMEOUT', '1800')),
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
    reasoning: str = ''   # 开启 thinking 时服务端分离出的推理文本（</think> 之前）
    inference_seconds: float = 0.   # 模型实际推理的时间，不含排队和加载模型；latency_seconds 包含这两项


class VLMClient:
    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig.from_env()
        self.runtime = VLMRuntime(self.config.base_url, self.config.model)
        # 与服务端 --max-batch-size 同源；超出的请求在客户端排队，不计入 HTTP 超时。
        self._slots = threading.BoundedSemaphore(self.runtime.max_num_seqs)
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
            raise VLMError(f'VLM HTTP {exc.code}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise VLMError(f'本机 VLM 服务请求失败: {exc}') from exc

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
               response_path: Path | None = None, enable_thinking: bool = False,
               thinking_token_budget: int | None = None, top_p: float | None = None,
               top_k: int | None = None, seed: int | None = None,
               image_labels: Sequence[str] | None = None, reasoning_effort: str | None = None) -> InferenceResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt 不能为空')
        if len(images) > self.config.max_images:
            raise ValueError(f'每次最多允许 {self.config.max_images} 张图片')
        labels = _check_image_labels(image_labels, len(images))
        tokens = self.config.max_tokens if max_tokens is None else max_tokens
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError('max_tokens 必须为正整数')
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature 必须为非负有限数值')
        if not isinstance(enable_thinking, bool):
            raise ValueError('enable_thinking 必须为布尔值')
        if thinking_token_budget is not None:
            if (not enable_thinking or isinstance(thinking_token_budget, bool)
                    or not isinstance(thinking_token_budget, int) or thinking_token_budget <= 0
                    or thinking_token_budget + 128 >= tokens):
                raise ValueError('thinking_token_budget 必须为正整数且为最终 JSON 留出至少 128 token')
            if not self.runtime.autostart:
                raise ValueError('受限 thinking 需由当前 runtime 启动的 exl3_server；外部服务模式不能保证预算生效')
        if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float))
                                  or not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError('top_p 必须在 (0,1] 内')
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError('top_k 必须为正整数')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError('seed 必须为非负整数')
        content = []
        for k, image in enumerate(images):
            if labels is not None:  # 标签紧贴在对应图片之前，与图片交替排列
                content.append({'type': 'text', 'text': labels[k]})
            content.append({'type': 'image_url', 'image_url': {'url': image}})
        content.append({'type': 'text', 'text': prompt})
        payload = {
            'model': self.config.model,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            'max_tokens': tokens, 'temperature': temperature, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': enable_thinking},
        }
        for key, value in (('top_p', top_p), ('top_k', top_k), ('seed', seed)):
            if value is not None:
                payload[key] = value
        if thinking_token_budget is not None:
            payload['thinking_token_budget'] = thinking_token_budget
        if reasoning_effort is not None and enable_thinking:
            payload['chat_template_kwargs']['reasoning_effort'] = reasoning_effort
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
            inference_started = time.monotonic()
            data = self._request('/chat/completions', payload)
            inference_seconds = time.monotonic() - inference_started
        if response_path is not None:
            from .common import write_json
            write_json(response_path, data)
        elapsed = time.monotonic() - started
        try:
            choice = data['choices'][0]
            text = choice['message']['content']
            reason = choice['finish_reason']
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError('VLM 服务返回了不完整的响应结构') from exc
        message = choice['message']
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        reasoning = reasoning if isinstance(reasoning, str) else ''
        if reason != 'stop':
            raise VLMError(f'推理未正常完成，finish_reason={reason}；可检查输出 token 上限',
                           raw_text=text if isinstance(text, str) else '', reasoning=reasoning)
        if not isinstance(text, str) or not text.strip():
            raise VLMError('模型返回空文本', reasoning=reasoning)
        return InferenceResult(text, data.get('usage', {}), elapsed, data.get('id', ''), reason, reasoning, inference_seconds)

    def infer(self, prompt: str, images: Sequence[str | Path] = (), *,
              system: str = '你是视觉分析助手。根据图片与任务给出准确回答，不猜测不可见内容。',
              max_tokens: int | None = None, temperature: float = 0.0,
              json_output: bool = False, output_schema: dict | None = None,
              image_labels: Sequence[str] | None = None, enable_thinking: bool = False,
              thinking_token_budget: int | None = None, reasoning_effort: str | None = None) -> InferenceResult:
        """通用节点接口；支持纯文本、单图、多图。不会自动重试昂贵的推理请求。

        enable_thinking / thinking_token_budget：开启 Qwen thinking 并限制推理 token；推理文本在 result.reasoning，
        结构化输出约束只作用于推理之后的回答。预算需要由本程序启动的服务（VLM_AUTOSTART=1）才能生效。

        image_labels：可选，与 images 等长的文字标签；给出时每张图前插入对应标签（文字/图片交替），
        便于模型把文字说明与具体图片对上。不给时保持原来的“全部图片 + 主提示词”结构。
        """
        if isinstance(images, (str, Path)):
            raise ValueError('images 请使用列表，例如 [image_path]')
        if len(images) > self.config.max_images:
            raise ValueError(f'每次最多允许 {self.config.max_images} 张图片')
        _check_image_labels(image_labels, len(images))
        prepared = [self._prepare_image(path) for path in images]
        return self._infer(prompt, prepared, system=system, max_tokens=max_tokens,
                           temperature=temperature, json_output=json_output, output_schema=output_schema,
                           image_labels=image_labels, enable_thinking=enable_thinking,
                           thinking_token_budget=thinking_token_budget, reasoning_effort=reasoning_effort)

