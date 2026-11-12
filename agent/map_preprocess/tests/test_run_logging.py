"""运行日志：节点进度、终端过滤、token 速度。"""
import contextlib
import io
import json
import logging
import tempfile
import time
import unittest
from pathlib import Path

from agent.map_preprocess.nodes.vlm_review import run_jobs
from agent.map_preprocess.run_logging import ConsoleHandler, NodeProgress, VLMMeter, format_duration


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def make_logger(name: str) -> tuple[logging.Logger, Capture]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    capture = Capture()
    logger.handlers = [capture]
    return logger, capture


class ProgressTests(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual([format_duration(v) for v in (5, 65, 3700)], ['5s', '1m05s', '1h01m'])

    def test_nodes_report_position_and_write_timing(self):
        logger, capture = make_logger('test.progress')
        progress = NodeProgress()
        first, second = progress.add('clean_road_graph'), progress.add('refresh_global_manifest')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'node_timing.json'
            progress.finish(logger, first, 'clean_road_graph', progress.begin(logger, first, 'clean_road_graph'), path)
            progress.finish(logger, second, 'refresh_global_manifest', progress.begin(logger, second, 'refresh_global_manifest'), path)
            saved = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual([n['node'] for n in saved['nodes']], ['clean_road_graph', 'refresh_global_manifest'])
        self.assertIn('[1/2] 开始 道路拓扑（clean_road_graph）', capture.lines[0])
        self.assertTrue(capture.lines[1].startswith('[1/2] 完成 道路拓扑'))
        self.assertIn('总用时', capture.lines[1])
        self.assertIn('流程结束', capture.lines[-1])

    def test_console_hides_detail_lines(self):
        logger = logging.getLogger('test.console')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers = [ConsoleHandler()]
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            logger.info('边覆盖/重叠 %s', {'edge_id': '0'})
            logger.info('过滤 %s', 'x')
            logger.info('[1/2] 开始')
        self.assertNotIn('边覆盖', errors.getvalue())
        self.assertIn('[1/2] 开始', errors.getvalue())


class MeterTests(unittest.TestCase):
    def test_rates_exclude_failures_and_use_inference_time(self):
        meter = VLMMeter()
        meter.add({'status': 'ok', 'usage': {'prompt_tokens': 100, 'completion_tokens': 50}, 'inference_seconds': 5., 'latency_seconds': 60.})
        meter.add({'status': 'error', 'usage': {'prompt_tokens': 100, 'completion_tokens': 50}})
        self.assertEqual((meter.done, meter.failed, meter.prompt_tokens, meter.completion_tokens), (2, 1, 100, 50))
        self.assertAlmostEqual(meter.per_request, 10.)
        self.assertLess(abs(meter.throughput-10.), 1.)

    def test_concurrent_requests_add_up_in_throughput(self):
        meter = VLMMeter()
        for _ in range(4):
            meter.add({'status': 'ok', 'usage': {'completion_tokens': 100}, 'inference_seconds': 10.})
        self.assertAlmostEqual(meter.per_request, 10.)
        self.assertLess(abs(meter.throughput-40.), 4.)

    def test_no_usage_gives_zero_rates(self):
        meter = VLMMeter()
        meter.add({'status': 'ok', 'inference_seconds': 2.})
        self.assertEqual((meter.throughput, meter.per_request), (0., 0.))
        self.assertEqual((meter.postfix()['成功'], meter.postfix()['失败']), (1, 0))

    def test_run_jobs_shows_speed_and_logs_summary(self):
        logger, capture = make_logger('test.jobs')
        def work(job):
            time.sleep(.01)
            return {'status': 'ok', 'usage': {'prompt_tokens': 10, 'completion_tokens': 20}, 'inference_seconds': .5}
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            run_jobs(work, [{}, {}], 2, 'demo', logger)
        self.assertIn('tok/s', errors.getvalue())
        self.assertTrue(any('VLM 请求 2/2' in line for line in capture.lines))
        self.assertTrue(any('输出 40.0 tok/s' in line for line in capture.lines))
        self.assertTrue(any(line.startswith('demo 汇总') and '输出 40 tokens' in line for line in capture.lines))


if __name__ == '__main__':
    unittest.main()
