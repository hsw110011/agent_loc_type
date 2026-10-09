"""首次推理才启动本机 INT8 服务；同一客户端的线程共享一份模型。"""
from __future__ import annotations

import atexit
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time
import urllib.request
from urllib.parse import urlparse
from datetime import datetime
from zoneinfo import ZoneInfo


class VLMRuntime:
    def __init__(self, base_url: str, model: str):
        self.base_url, self.model = base_url, model
        self.autostart = os.getenv('VLM_AUTOSTART', '1') == '1'
        self.start_timeout = float(os.getenv('VLM_START_TIMEOUT', '600'))
        if self.start_timeout <= 0:
            raise ValueError('VLM_START_TIMEOUT 必须为正数')
        self.log_dir = Path(__file__).parent / 'log'
        self.process = None
        self.log_path = None
        self._lock_file = None
        self._mutex = threading.RLock()
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        atexit.register(self.close)

    def _server_command(self, port):
        """在独立进程中导入量化器和 vLLM，Agent 不提前加载 VLM。"""
        root = Path(__file__).resolve().parent.parent
        from .common import resolve_path
        python = resolve_path(os.getenv('PROJECT_PYTHON', str(root / '.venv-vllm/bin/python')))
        model_path = resolve_path(os.getenv('VLM_MODEL_PATH', str(root / 'qwen_3.5_9b')))
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f'Python 不可用: {python}')
        if not (model_path / 'config.json').is_file():
            raise ValueError(f'模型配置不存在: {model_path}/config.json')
        bootstrap = '''import json, os, sys
from torchao.core.config import config_to_dict
from torchao.quantization import Int8WeightOnlyConfig, ModuleFqnToConfig
config = config_to_dict(ModuleFqnToConfig({
    '_default': Int8WeightOnlyConfig(),
    r're:(?:.*\\.)?visual\\..*': None,
}))
overrides = json.dumps({'quantization_config_dict_json': json.dumps(config)})
os.execv(sys.executable, [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                        *sys.argv[1:], '--hf-overrides', overrides])
'''
        command = [str(python), '-c', bootstrap,
                   '--model', str(model_path), '--served-model-name', self.model,
                   '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                   '--quantization', 'torchao', '--max-model-len', os.getenv('VLM_MAX_MODEL_LEN', '17584'),
                   '--gpu-memory-utilization', os.getenv('VLM_GPU_MEMORY_UTILIZATION', '0.65'),
                   '--max-num-seqs', os.getenv('VLM_MAX_NUM_SEQS', '2'),
                   '--max-num-batched-tokens', os.getenv('VLM_MAX_BATCHED_TOKENS', '2048'),
                   '--mm-processor-cache-gb', '0', '--enforce-eager',
                   '--limit-mm-per-prompt', '{"image":4,"video":0}',
                   '--mm-processor-kwargs', '{"size":{"longest_edge":1048576,"shortest_edge":65536}}',
                   '--structured-outputs-config', '{"backend":"xgrammar","disable_fallback":true}',
                   '--reasoning-parser', 'qwen3', '--generation-config', 'vllm']
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1')
        base_lib = Path(sys.base_prefix) / 'lib'
        if (base_lib / 'libstdc++.so.6').is_file():
            env['LD_LIBRARY_PATH'] = str(base_lib) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        return command, env

    def _ready(self):
        try:
            with self._http.open(self.base_url.rstrip('/') + '/models', timeout=2) as response:
                data = json.load(response)
            return self.model in [item.get('id') for item in data.get('data', [])]
        except (OSError, ValueError):
            return False

    def ensure_started(self):
        if not self.autostart:
            return  # 显式外部服务模式，由调用者负责部署与生命周期。
        with self._mutex:
            if self.process is not None:
                if self.process.poll() is None and self._ready():
                    return
                raise RuntimeError(f'VLM 服务已退出或不可用；查看 {self.log_path}')
            parsed = urlparse(self.base_url)
            if parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.path.rstrip('/') != '/v1':
                raise ValueError('自动启动仅支持 127.0.0.1/localhost 的 /v1 地址')
            port = parsed.port or 80
            self.log_dir.mkdir(parents=True, exist_ok=True)
            project_key = hashlib.sha256(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]
            self._lock_file = (Path(tempfile.gettempdir()) / f'hsw_vlm_{project_key}_{port}.lock').open('a')
            try:
                try:
                    fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f'端口 {port} 已由另一个 VLM 任务管理；请共享同一个 VLMClient') from exc
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        probe.bind(('127.0.0.1', port))
                    except OSError as exc:
                        raise RuntimeError(f'端口 {port} 已占用，无法确认已有服务为 INT8；请先关闭旧服务，或显式设置 VLM_AUTOSTART=0') from exc
                # Node 2 的模型已离开作用域，释放其缓存，不在 agent 进程导入 VLM。
                gc.collect()
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
                self.log_path = self.log_dir / f'{timestamp}_vlm.log'
                command, env = self._server_command(port)
                if env.get('VLM_API_KEY'):
                    raise ValueError('自动启动服务暂不配置认证；请移除 VLM_API_KEY 或使用外部服务模式')
                with self.log_path.open('a') as log:
                    self.process = subprocess.Popen(
                        command,
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + self.start_timeout
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f'INT8 服务启动失败；查看 {self.log_path}')
                    if self._ready():
                        return
                    time.sleep(0.5)
                raise RuntimeError(f'INT8 服务启动超时；查看 {self.log_path}')
            except BaseException:
                self.close()
                raise

    def close(self):
        """仅停止本实例启动的进程组；可重复调用。"""
        with self._mutex:
            if self.process is not None:
                process, self.process = self.process, None
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    process.wait()
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None
