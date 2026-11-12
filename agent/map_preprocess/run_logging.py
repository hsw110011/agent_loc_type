"""运行日志：终端输出、节点进度与用时、VLM token 速度。详细记录仍写入任务目录的 state/run.log。"""
from __future__ import annotations

import glob
import logging
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .io import write_json

NODE_LABELS = {'load_image_and_resolution': '读图与分辨率', 'classify_scene_type': 'VLM 场景判断', 'extract_road_graph': 'MaGRoad 道路提取',
               'clean_road_graph': '道路拓扑', 'build_blocks_and_groups': '布块与去重', 'compute_block_features': '块级语义/高度',
               'vlm_review_blocks': 'VLM 退化复核', 'vlm_expand_blocks': 'VLM 退化扩展', 'fuse_degraded_patches': '退化区域融合', 'recommend_localization': 'VLM 定位方式推荐', 'refresh_global_manifest': '汇总 manifest'}
# 这些明细只写 run.log，不进终端
QUIET_MESSAGES = ('边覆盖/重叠', '过滤 ', 'cross_edge_dedup', 'VLM 请求', '%s 推理完成')


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds//60}m{seconds%60:02d}s'
    return f'{seconds//3600}h{seconds%3600//60:02d}m'


def cpu_temperature() -> float | None:
    """CPU 温度（°C）；读不到时返回 None。"""
    for name_path in glob.glob('/sys/class/hwmon/hwmon*/name'):
        try:
            if Path(name_path).read_text().strip() in ('coretemp', 'k10temp'):
                return int(Path(name_path).with_name('temp1_input').read_text())/1000
        except (OSError, ValueError):
            continue
    return None


def system_status() -> str:
    parts = []
    temperature = cpu_temperature()
    if temperature is not None:
        parts.append(f'CPU {temperature:.0f}°C')
    parts.append(f'内存峰值 {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.1f} GB')
    return ' | '.join(parts)


class ConsoleHandler(logging.Handler):
    """写到终端，不打断 tqdm 进度条。"""

    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S'))
        self.addFilter(lambda record: not str(record.msg).startswith(QUIET_MESSAGES))

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


@dataclass
class NodeProgress:
    """流程的节点清单和用时；每个节点结束时写入 state/node_timing.json，中途失败也能看到。"""
    names: list[str] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    started: float | None = None

    def add(self, name: str) -> int:
        self.names.append(name)
        return len(self.names)

    def begin(self, logger: logging.Logger, index: int, name: str) -> float:
        now = time.monotonic()
        self.started = self.started if self.started is not None else now
        logger.info('[%d/%d] 开始 %s（%s）', index, len(self.names), NODE_LABELS.get(name, name), name)
        return now

    def finish(self, logger: logging.Logger, index: int, name: str, began: float, path: Path) -> None:
        now = time.monotonic()
        self.timings.append({'node': name, 'seconds': round(now-began, 1)})
        write_json(path, {'total_seconds': round(now-self.started, 1), 'nodes': self.timings})
        logger.info('[%d/%d] 完成 %s，用时 %s | 总用时 %s | %s', index, len(self.names), NODE_LABELS.get(name, name),
                    format_duration(now-began), format_duration(now-self.started), system_status())
        if index == len(self.names):
            rows = ' | '.join(f"{NODE_LABELS.get(t['node'], t['node'])} {format_duration(t['seconds'])}" for t in self.timings)
            logger.info('流程结束，总用时 %s：%s', format_duration(now-self.started), rows)


class VLMMeter:
    """统计一批 VLM 请求的 token 和速度。

    tok/s 是整批吞吐（已完成的输出 token / 从第一个请求开始推理到现在的时间），不含加载模型的等待；
    每路 tok/s 是单个请求在推理期间的平均速度，并发时吞吐约等于每路速度 × 并发数。
    """

    def __init__(self):
        self.done = self.failed = self.prompt_tokens = self.completion_tokens = 0
        self.measured_tokens = 0
        self.inference_seconds = 0.
        self.first_start: float | None = None

    def add(self, record: dict) -> None:
        self.done += 1
        if record.get('status') != 'ok':
            self.failed += 1
            return
        usage = record.get('usage') or {}
        seconds = record.get('inference_seconds') or record.get('latency_seconds') or 0.
        self.prompt_tokens += int(usage.get('prompt_tokens') or 0)
        tokens = usage.get('completion_tokens')
        self.completion_tokens += int(tokens or 0)
        if tokens is None or seconds <= 0:
            return
        self.measured_tokens += int(tokens)
        self.inference_seconds += seconds
        began = time.monotonic()-seconds
        self.first_start = began if self.first_start is None else min(self.first_start, began)

    @property
    def throughput(self) -> float:
        elapsed = time.monotonic()-self.first_start if self.first_start is not None else 0.
        return self.measured_tokens/elapsed if elapsed > 0 else 0.

    @property
    def per_request(self) -> float:
        return self.measured_tokens/self.inference_seconds if self.inference_seconds > 0 else 0.

    def postfix(self) -> dict:
        return {'tok/s': f'{self.throughput:.1f}', '每路': f'{self.per_request:.1f}', '输出': self.completion_tokens,
                '成功': self.done-self.failed, '失败': self.failed}

    def summary(self) -> str:
        return (f'{self.done} 个请求（成功 {self.done-self.failed}，失败 {self.failed}），输入 {self.prompt_tokens} / 输出 {self.completion_tokens} tokens，'
                f'吞吐 {self.throughput:.1f} tok/s，每路 {self.per_request:.1f} tok/s')
