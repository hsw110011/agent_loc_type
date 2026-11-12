"""VLM 批量推理共用部分：进度条、thinking 配置语义、失败记录。"""
from __future__ import annotations

import contextlib
import io
import threading
import unittest

from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.nodes.vlm_review import failure_fields, run_jobs
from agent.vlm.client import VLMError


class RunJobsTests(unittest.TestCase):
    def test_results_keep_job_order_when_later_jobs_finish_first(self):
        second_done = threading.Event()

        def work(job):
            if job['id'] == 0:
                self.assertTrue(second_done.wait(5))
            else:
                second_done.set()
            return {'id': job['id']}

        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            records = run_jobs(work, [{'id': 0}, {'id': 1}], 2, 'demo')
        self.assertEqual([r['id'] for r in records], [0, 1])
        self.assertIn('demo', errors.getvalue())
        self.assertIn('2/2', errors.getvalue())

    def test_job_error_propagates(self):
        def work(job):
            raise ValueError('boom')

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
            run_jobs(work, [{'id': 0}], 1, 'demo')


class ThinkingConfigTests(unittest.TestCase):
    def test_thinking_is_off_by_default(self):
        config = GlobalPreprocessConfig()
        self.assertEqual(config.scene_vlm_thinking_budget, 0)
        self.assertEqual(config.vlm_review_thinking_budget, 0)
        self.assertEqual(config.vlm_expansion_thinking_budget, 0)

    def test_budget_values(self):
        for name in ('vlm_review_thinking_budget', 'vlm_expansion_thinking_budget'):
            self.assertEqual(getattr(GlobalPreprocessConfig(**{name: 0}), name), 0)
            for bad in (-1, 1.5, True):
                with self.assertRaises(ValueError, msg=(name, bad)):
                    GlobalPreprocessConfig(**{name: bad})


class FailureFieldsTests(unittest.TestCase):
    def test_keeps_generated_text_only_when_present(self):
        record = failure_fields(VLMError('超限', raw_text='{', reasoning='想'))
        self.assertEqual((record['status'], record['raw_text'], record['reasoning']), ('error', '{', '想'))
        self.assertEqual(set(failure_fields(ValueError('x'))), {'status', 'error'})


if __name__ == '__main__':
    unittest.main()
