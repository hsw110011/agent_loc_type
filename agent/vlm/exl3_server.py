#!/usr/bin/env python3
"""本地 EXL3 Qwen3.8-27B + DFlash2 的 OpenAI 兼容服务，只实现 agent/vlm/client.py 需要的接口。

由 runtime.py 用 Agent 同一个环境（项目 .venv，ExLlamaV3 已编译安装）以脚本路径启动为独立进程，
不导入 agent 包；模块顶层只用标准库，请求解析与 thinking 阶段逻辑可离线测试。

  GET  /v1/models            服务名与上下文上限（runtime 探活）
  POST /v1/chat/completions  非流式；文字/图片交替的多图消息、chat_template_kwargs、
                             response_format（json_schema / json_object，llguidance 约束）、
                             thinking_token_budget（按请求隔离的 thinking 上限）

thinking 分两段生成：第一段遇 </think> 或达到预算即停；预算用尽时补上 </think> 强制闭合，
第二段在其后生成回答，结构化约束只作用于回答。推理文本在 message.reasoning_content。
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 64 * 1024 * 1024
LOOPBACK = {'127.0.0.1', 'localhost', '::1'}
TEMPLATE_KWARGS = {'enable_thinking', 'reasoning_effort', 'preserve_thinking'}
THINK_END = '</think>'
# 模板为每张图输出的占位；ExLlamaV3 的图片嵌入自带 vision_start/end，整段替换为嵌入别名。
IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'


class RequestError(ValueError):
    """请求不合法，返回 HTTP 400。"""


# ---------------------------------------------------------------- 请求解析（纯函数）

def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RequestError(f'{name} 必须为正整数')
    return value


def _image_bytes(url) -> bytes:
    """只接受 data URL，服务端不访问任何外部地址。"""
    if not isinstance(url, str) or not url.startswith('data:image/') or ',' not in url:
        raise RequestError('图片仅支持 data:image/...;base64 URL')
    header, encoded = url.split(',', 1)
    if not header.endswith(';base64'):
        raise RequestError('图片 data URL 必须为 base64 编码')
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError(f'图片 base64 解码失败: {exc}') from exc


def _content(content, role):
    """OpenAI content → 模板 content；image_url 换成 {"type": "image"}，图片按出现顺序收集。"""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise RequestError(f'{role} 消息的 content 必须为字符串或数组')
    parts, images = [], []
    for part in content:
        kind = part.get('type') if isinstance(part, dict) else None
        if kind == 'text':
            if not isinstance(part.get('text'), str):
                raise RequestError('text 片段缺少字符串 text')
            parts.append({'type': 'text', 'text': part['text']})
        elif kind == 'image_url':
            if role != 'user':
                raise RequestError('只有 user 消息可以包含图片')
            url = part.get('image_url')
            images.append(_image_bytes(url.get('url') if isinstance(url, dict) else url))
            parts.append({'type': 'image'})
        else:
            raise RequestError(f'不支持的 content 片段类型: {kind!r}')
    return parts, images


def parse_request(body, *, model_name: str, max_images: int) -> dict:
    """校验 /v1/chat/completions 请求体，返回与模型无关的生成规格；图片仍为原始字节。"""
    if not isinstance(body, dict):
        raise RequestError('请求体必须为 JSON 对象')
    if body.get('stream'):
        raise RequestError('不支持流式输出')
    if body.get('n', 1) != 1:
        raise RequestError('仅支持 n=1')
    if body.get('tools'):
        raise RequestError('不支持 tools')
    if body.get('model', model_name) != model_name:
        raise RequestError(f"服务未提供模型 {body.get('model')!r}")
    raw = body.get('messages')
    if not isinstance(raw, list) or not raw:
        raise RequestError('messages 必须为非空数组')
    messages, images = [], []
    for message in raw:
        if not isinstance(message, dict) or message.get('role') not in {'system', 'user', 'assistant'}:
            raise RequestError('消息 role 只能为 system / user / assistant')
        content, found = _content(message.get('content'), message['role'])
        messages.append({'role': message['role'], 'content': content})
        images += found
    if len(images) > max_images:
        raise RequestError(f'每次最多允许 {max_images} 张图片')

    kwargs = body.get('chat_template_kwargs') or {}
    if not isinstance(kwargs, dict) or set(kwargs) - TEMPLATE_KWARGS:
        raise RequestError(f'chat_template_kwargs 只支持 {sorted(TEMPLATE_KWARGS)}')
    thinking = kwargs.get('enable_thinking', True)  # 与模板一致：未指定时开启 thinking
    if not isinstance(thinking, bool):
        raise RequestError('enable_thinking 必须为布尔值')
    kwargs = {**kwargs, 'enable_thinking': thinking}

    max_tokens = _positive_int(body.get('max_tokens', 4096), 'max_tokens')
    budget = body.get('thinking_token_budget')
    if budget is not None:
        _positive_int(budget, 'thinking_token_budget')
        if not thinking:
            raise RequestError('thinking_token_budget 需要 enable_thinking=true')
        if budget >= max_tokens:
            raise RequestError('thinking_token_budget 必须小于 max_tokens，为回答留出 token')

    temperature = body.get('temperature', 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not math.isfinite(temperature) or temperature < 0:
        raise RequestError('temperature 必须为非负有限数值')
    top_p = body.get('top_p', 1.0)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise RequestError('top_p 必须在 (0,1] 内')
    top_k = body.get('top_k', 0)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise RequestError('top_k 必须为非负整数（0 = 不限）')
    seed = body.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise RequestError('seed 必须为非负整数')

    schema = None
    response_format = body.get('response_format') or {'type': 'text'}
    kind = response_format.get('type') if isinstance(response_format, dict) else None
    if kind == 'json_schema':
        schema = (response_format.get('json_schema') or {}).get('schema')
        if not isinstance(schema, dict):
            raise RequestError('response_format.json_schema.schema 必须为 JSON Schema 对象')
    elif kind == 'json_object':
        schema = {'type': 'object'}
    elif kind != 'text':
        raise RequestError(f'不支持的 response_format: {kind!r}')

    return {'messages': messages, 'images': images, 'template_kwargs': kwargs, 'thinking': thinking,
            'max_tokens': max_tokens, 'budget': budget, 'temperature': float(temperature),
            'top_p': float(top_p), 'top_k': top_k, 'seed': seed, 'schema': schema}


# ---------------------------------------------------------------- thinking 阶段（纯函数）

def think_limit(max_tokens: int, budget: int | None) -> int:
    """第一段最多生成的 token 数；不设预算时与 max_tokens 相同。"""
    return max_tokens if budget is None else min(budget, max_tokens)


def after_think(*, eos_reason, closed: bool, think_tokens: int, max_tokens: int, budget: int | None):
    """第一段结束后的去向：('answer', 闭合文本, 回答 token 上限) 或 ('finish', finish_reason)。

    closed：第一段因 </think> 结束；达到预算上限时补 "\\n</think>\\n\\n" 强制闭合。
    </think> 计入 completion token。
    """
    if closed:
        suffix = THINK_END + '\n\n'
    elif eos_reason == 'max_new_tokens' and budget is not None and think_tokens >= budget:
        suffix = '\n' + THINK_END + '\n\n'
    elif eos_reason == 'max_new_tokens':
        return ('finish', 'length')     # 没有预算，thinking 用尽了全部 max_tokens
    else:
        return ('finish', 'stop')       # thinking 中途直接结束（EOS），没有回答
    remaining = max_tokens - think_tokens - 1
    if remaining <= 0:
        return ('finish', 'length')
    return ('answer', suffix, remaining)


def finish_reason(eos_reason) -> str:
    return 'length' if eos_reason == 'max_new_tokens' else 'stop'


# ---------------------------------------------------------------- 模型服务

class Server:
    def __init__(self, args):
        import torch
        from exllamav3 import Cache, CacheLayer_quant, Config, Generator, LLGuidanceFilter, Model, Tokenizer
        from exllamav3.generator.sampler import ComboSampler
        from exllamav3 import Job
        from PIL import Image, ImageOps
        self.torch, self.Job, self.ComboSampler, self.LLGuidanceFilter = torch, Job, ComboSampler, LLGuidanceFilter
        self.Image, self.ImageOps = Image, ImageOps
        self.args, self.model_name = args, args.served_model_name

        started = time.time()
        config = Config.from_directory(args.target)
        if config.vision_pp is not None:
            config.vision_pp.max_pixels = args.max_pixels   # 与客户端 VLM_MAX_IMAGE_PIXELS 一致
        draft_model = draft_cache = None
        history = 0
        if args.draft:
            draft_model = Model.from_config(Config.from_directory(args.draft))
            history = draft_model.caps.get('default_draft_size', 4)
        model = Model.from_config(config)
        quant = {'layer_type': CacheLayer_quant, 'k_bits': args.cq, 'v_bits': args.cq} if args.cq else {}
        cache = Cache(model, max_num_tokens=args.cache_tokens, max_history=history,
                      max_batch_size=args.max_batch_size, **quant)
        model.load(progressbar=False)
        if draft_model is not None:
            draft_cache = Cache(draft_model, max_num_tokens=args.cache_tokens,
                                max_batch_size=args.max_batch_size, **quant)
            draft_model.load(progressbar=False)
        self.vision = Model.from_config(config, component='vision')
        self.vision.load(progressbar=False)
        self.tokenizer = Tokenizer.from_config(config)
        self.generator = Generator(model, cache, self.tokenizer, draft_model=draft_model, draft_cache=draft_cache,
                                   max_batch_size=args.max_batch_size)

        self.think_end = self.tokenizer.single_id(THINK_END)
        stops = list(getattr(config, 'eos_token_id_list', None) or [])
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in stops:
            stops.append(self.tokenizer.eos_token_id)
        self.stop_ids = stops
        self.queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()
        free, total = torch.cuda.mem_get_info()
        print(f'[exl3] loaded in {time.time() - started:.0f}s; draft={bool(args.draft)}, '
              f'cache {args.cache_tokens} tokens cq{args.cq}, batch {args.max_batch_size}; '
              f'VRAM {(total - free) / 1e9:.2f}/{total / 1e9:.2f} GB', flush=True)

    # ---- 对外：HTTP 线程调用，阻塞到请求完成

    def submit(self, spec: dict) -> dict:
        spec['pil'] = [self._image(data) for data in spec.pop('images')]
        if spec['schema'] is not None:  # 语法错误在 HTTP 线程直接返回 400，不占用生成线程
            try:
                self._filter(spec['schema'])
            except ValueError as exc:
                raise RequestError(f'response_format 无法编译为 llguidance 语法: {exc}') from exc
        holder = {'event': threading.Event(), 'result': None, 'error': None}
        self.queue.put((spec, holder))
        holder['event'].wait()
        if holder['error'] is not None:
            raise holder['error']
        return holder['result']

    def _image(self, data: bytes):
        try:
            with self.Image.open(io.BytesIO(data)) as source:
                return self.ImageOps.exif_transpose(source).convert('RGB')
        except Exception as exc:  # noqa: BLE001  PIL 对坏图抛出多种异常
            raise RequestError(f'无法解码图片: {exc}') from exc

    def _filter(self, schema):
        return self.LLGuidanceFilter(self.tokenizer, json_schema=schema, eos_after_completed=True)

    # ---- 生成线程：唯一使用 GPU 的线程，交替接收新请求和推进生成器

    def _loop(self):
        jobs = {}   # identifier -> request state
        while True:
            block = not self.generator.num_remaining_jobs()
            while True:
                try:
                    spec, holder = self.queue.get(block=block)
                except queue.Empty:
                    break
                block = False
                try:
                    self._start(spec, holder, jobs)
                except Exception as exc:  # noqa: BLE001
                    self._fail(holder, exc)
            if not self.generator.num_remaining_jobs():
                continue
            try:
                results = self.generator.iterate()
            except Exception as exc:  # noqa: BLE001  生成器异常时终止全部在途请求，服务继续
                for state in list(jobs.values()):
                    self.generator.cancel(state['job'])
                    self._fail(state['holder'], RuntimeError(f'生成失败: {type(exc).__name__}: {exc}'))
                jobs.clear()
                continue
            for result in results:
                state = jobs.get(result.get('identifier'))
                if state is not None and result.get('stage') == 'streaming':
                    self._collect(state, result, jobs)

    def _start(self, spec, holder, jobs):
        embeddings = [self.vision.get_image_embeddings(tokenizer=self.tokenizer, image=image) for image in spec['pil']]
        rendered = self.tokenizer.hf_render_chat_template(spec['messages'], add_generation_prompt=True,
                                                          **spec['template_kwargs'])
        if rendered.count(IMAGE_PLACEHOLDER) != len(embeddings):
            raise RequestError('模板中的图片占位符数量与图片数量不一致')
        for embedding in embeddings:
            rendered = rendered.replace(IMAGE_PLACEHOLDER, embedding.text_alias, 1)
        prompt = self.tokenizer.encode(rendered, encode_special_tokens=True, embeddings=embeddings or None)
        if prompt.shape[-1] + spec['max_tokens'] > self.args.max_model_len:
            raise RequestError(f"prompt {prompt.shape[-1]} + max_tokens {spec['max_tokens']} "
                               f'超过 max_model_len {self.args.max_model_len}')
        state = {'spec': spec, 'holder': holder, 'embeddings': embeddings, 'prompt': prompt,
                 'think_ids': [], 'answer_ids': [], 'closed_think': False, 'stats': [], 'started': time.monotonic()}
        if spec['thinking']:
            self._enqueue(state, jobs, 'think', prompt, think_limit(spec['max_tokens'], spec['budget']),
                          stops=[self.think_end] + self.stop_ids, constrained=False)
        else:
            self._enqueue(state, jobs, 'answer', prompt, spec['max_tokens'], stops=self.stop_ids, constrained=True)

    def _enqueue(self, state, jobs, phase, input_ids, limit, *, stops, constrained):
        spec = state['spec']
        identifier = uuid.uuid4().hex
        sampler = self.ComboSampler(temperature=spec['temperature'], top_k=spec['top_k'], top_p=spec['top_p'])
        filters = [self._filter(spec['schema'])] if constrained and spec['schema'] is not None else None
        job = self.Job(input_ids=input_ids, max_new_tokens=limit, sampler=sampler, seed=spec['seed'],
                       stop_conditions=stops, filters=filters, embeddings=state['embeddings'] or None,
                       identifier=identifier)
        state.update({'phase': phase, 'job': job, 'ids': []})
        jobs[identifier] = state
        self.generator.enqueue(job)

    def _collect(self, state, result, jobs):
        ids = result.get('token_ids')
        if ids is not None:
            state['ids'] += ids.flatten().tolist()
        if not result.get('eos'):
            return
        jobs.pop(result['identifier'], None)
        reason = result.get('eos_reason')
        held = (result.get('held') or {}).get('token_ids')
        if held is not None:
            held = held.flatten().tolist()
            state['ids'] += held[:-1] if reason == 'stop_token' else held   # 停止 token 本身不进入输出
        state['stats'].append({key: result.get(key) for key in (
            'prompt_tokens', 'new_tokens', 'cached_tokens', 'time_prefill', 'time_generate',
            'accepted_draft_tokens', 'rejected_draft_tokens')})
        spec = state['spec']
        try:
            if state['phase'] == 'think':
                state['think_ids'] = state['ids']
                closed = reason == 'stop_token' and result.get('eos_triggering_token_id') == self.think_end
                state['closed_think'] = closed
                step = after_think(eos_reason=reason, closed=closed, think_tokens=len(state['think_ids']),
                                   max_tokens=spec['max_tokens'], budget=spec['budget'])
                if step[0] == 'finish':
                    return self._finish(state, step[1])
                state['closed_think'] = True
                suffix = self.tokenizer.encode(step[1], encode_special_tokens=True)
                think = self.torch.tensor([state['think_ids']], dtype=self.torch.long)
                answer_input = self.torch.cat([state['prompt'], think, suffix], dim=-1)
                return self._enqueue(state, jobs, 'answer', answer_input, step[2], stops=self.stop_ids, constrained=True)
            state['answer_ids'] = state['ids']
            self._finish(state, finish_reason(reason))
        except Exception as exc:  # noqa: BLE001
            self._fail(state['holder'], exc)

    def _decode(self, ids) -> str:
        if not ids:
            return ''
        return self.tokenizer.decode(self.torch.tensor(ids, dtype=self.torch.long), decode_special_tokens=False)

    def _finish(self, state, reason):
        spec = state['spec']
        completion = len(state['think_ids']) + int(state['closed_think']) + len(state['answer_ids'])
        prompt_tokens = int(state['prompt'].shape[-1])
        reasoning = self._decode(state['think_ids']).strip() if spec['thinking'] else None
        state['holder']['result'] = {
            'id': f'chatcmpl-{uuid.uuid4().hex[:24]}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': self.model_name,
            'choices': [{'index': 0, 'finish_reason': reason, 'message': {
                'role': 'assistant', 'content': self._decode(state['answer_ids']).strip(),
                'reasoning_content': reasoning}}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion,
                      'total_tokens': prompt_tokens + completion},
            'exl3': {'seconds': round(time.monotonic() - state['started'], 3), 'phases': state['stats']},
        }
        state['holder']['event'].set()

    @staticmethod
    def _fail(holder, exc):
        if not holder['event'].is_set():
            holder['error'] = exc
            holder['event'].set()


SERVER: Server | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):  # noqa: A003
        print(f'[http] {fmt % args}', flush=True)

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message):
        self._send(code, {'error': {'message': message, 'type': 'invalid_request_error' if code == 400 else 'server_error'}})

    def do_GET(self):  # noqa: N802
        if self.path == '/v1/models':
            return self._send(200, {'object': 'list', 'data': [{
                'id': SERVER.model_name, 'object': 'model', 'owned_by': 'local',
                'max_model_len': SERVER.args.max_model_len}]})
        if self.path in {'/health', '/v1/health'}:
            return self._send(200, {'status': 'ok'})
        self._error(404, 'not found')

    def do_POST(self):  # noqa: N802
        if self.path != '/v1/chat/completions':
            return self._error(404, 'not found')
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > MAX_BODY:
            return self._error(400, f'请求体大小必须在 1..{MAX_BODY} 字节之间')
        try:
            body = json.loads(self.rfile.read(length))
            spec = parse_request(body, model_name=SERVER.model_name, max_images=SERVER.args.max_images)
            return self._send(200, SERVER.submit(spec))
        except (RequestError, json.JSONDecodeError) as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error(500, f'{type(exc).__name__}: {exc}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--target', required=True, help='目标模型 EXL3 目录')
    parser.add_argument('--draft', default='', help='DFlash2 草稿模型 EXL3 目录；留空关闭投机解码')
    parser.add_argument('--served-model-name', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--cache-tokens', type=int, required=True, help='KV cache 总 token 数（所有并发序列共享）')
    parser.add_argument('--cq', type=int, default=3, help='KV cache 量化位数；0 = fp16')
    parser.add_argument('--max-batch-size', type=int, default=1, help='同时生成的序列数')
    parser.add_argument('--max-model-len', type=int, required=True, help='单请求 prompt + max_tokens 上限')
    parser.add_argument('--max-images', type=int, default=4)
    parser.add_argument('--max-pixels', type=int, default=1048576, help='视觉预处理的单图像素上限')
    return parser


def main(argv=None):
    global SERVER
    args = build_parser().parse_args(argv)
    if args.host not in LOOPBACK:
        raise SystemExit('只允许监听本机回环地址')
    for name in ('cache_tokens', 'max_batch_size', 'max_model_len', 'max_images', 'max_pixels'):
        if getattr(args, name) <= 0:
            raise SystemExit(f'--{name.replace("_", "-")} 必须为正整数')
    if args.max_model_len > args.cache_tokens:
        raise SystemExit('--max-model-len 不能超过 --cache-tokens')
    SERVER = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f'[exl3] listening on http://{args.host}:{args.port}/v1', flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
