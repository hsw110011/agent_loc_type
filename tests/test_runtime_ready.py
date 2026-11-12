"""服务探活：已启动的 VLM 服务忙于推理时 /models 可能短暂超时，不应被误判为服务退出。不启动真实服务。"""
import unittest
from unittest import mock

from agent.vlm.runtime import VLMRuntime


class FakeProcess:
    def __init__(self, alive: bool = True):
        self.alive = alive

    def poll(self):
        return None if self.alive else 1


class RuntimeReadyTests(unittest.TestCase):
    def runtime(self, alive: bool, answers: list[bool]) -> VLMRuntime:
        runtime = VLMRuntime('http://127.0.0.1:8000/v1', 'model')
        runtime.autostart, runtime.ready_timeout = True, 30
        runtime.process = FakeProcess(alive)
        runtime._ready = mock.Mock(side_effect=answers+[True]*10)
        return runtime

    def test_busy_service_is_waited_for(self):
        runtime = self.runtime(True, [False, False])
        with mock.patch('agent.vlm.runtime.time.sleep'):
            runtime.ensure_started()
        self.assertEqual(runtime._ready.call_count, 3)
        runtime.process = None  # 假进程，不让 close() 去终止

    def test_dead_service_fails_fast(self):
        runtime = self.runtime(False, [])
        with self.assertRaisesRegex(RuntimeError, '已退出或不可用'):
            runtime.ensure_started()
        runtime._ready.assert_not_called()
        runtime.process = None

    def test_unresponsive_service_gives_up_after_timeout(self):
        runtime = self.runtime(True, [False]*100)
        runtime.ready_timeout = 0
        with mock.patch('agent.vlm.runtime.time.sleep'), self.assertRaisesRegex(RuntimeError, '已退出或不可用'):
            runtime.ensure_started()
        runtime.process = None


if __name__ == '__main__':
    unittest.main()
