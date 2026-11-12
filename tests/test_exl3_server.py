"""EXL3 服务的请求解析、thinking 分段和启动命令；不加载模型、不导入 ExLlamaV3。"""
import base64
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

from agent.vlm.runtime import VLMRuntime

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('exl3_server', ROOT/'agent/vlm/exl3_server.py')
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)

PNG = 'data:image/png;base64,' + base64.b64encode(b'not-really-a-png').decode()


def body(**extra):
    return {'model': 'm', 'max_tokens': 2048,
            'messages': [{'role': 'system', 'content': '系统'},
                         {'role': 'user', 'content': [{'type': 'text', 'text': '图1'}, {'type': 'image_url', 'image_url': {'url': PNG}},
                                                      {'type': 'text', 'text': '图2'}, {'type': 'image_url', 'image_url': {'url': PNG}},
                                                      {'type': 'text', 'text': '主提示'}]}],
            **extra}


def parse(**extra):
    return server.parse_request(body(**extra), model_name='m', max_images=4)


class ParseRequestTests(unittest.TestCase):
    def test_labels_and_images_keep_interleaved_order(self):
        spec = parse()
        self.assertEqual([p['type'] for p in spec['messages'][1]['content']], ['text', 'image', 'text', 'image', 'text'])
        self.assertEqual(spec['images'], [b'not-really-a-png'] * 2)
        self.assertEqual(spec['messages'][0], {'role': 'system', 'content': '系统'})

    def test_thinking_defaults_on_like_template_and_budget_passthrough(self):
        self.assertTrue(parse()['thinking'])
        spec = parse(chat_template_kwargs={'enable_thinking': True}, thinking_token_budget=512)
        self.assertEqual((spec['budget'], spec['template_kwargs']), (512, {'enable_thinking': True}))
        self.assertFalse(parse(chat_template_kwargs={'enable_thinking': False})['thinking'])

    def test_response_format_maps_to_schema(self):
        schema = {'type': 'object', 'properties': {'a': {'type': 'integer'}}}
        fmt = {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': schema, 'strict': True}}
        self.assertEqual(parse(response_format=fmt)['schema'], schema)
        self.assertEqual(parse(response_format={'type': 'json_object'})['schema'], {'type': 'object'})
        self.assertIsNone(parse()['schema'])

    def test_invalid_requests_rejected(self):
        remote = body()
        remote['messages'][1]['content'][1]['image_url']['url'] = 'http://example.com/a.png'
        cases = [dict(stream=True), dict(model='other'), dict(max_tokens=0), dict(temperature=-1),
                 dict(top_p=0), dict(seed=-1), dict(response_format={'type': 'grammar'}),
                 dict(chat_template_kwargs={'enable_thinking': False}, thinking_token_budget=10),
                 dict(chat_template_kwargs={'enable_thinking': True}, thinking_token_budget=2048),
                 dict(chat_template_kwargs={'add_vision_id': True})]
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(server.RequestError):
                parse(**extra)
        with self.assertRaises(server.RequestError):
            server.parse_request(remote, model_name='m', max_images=4)
        with self.assertRaises(server.RequestError):
            server.parse_request(body(), model_name='m', max_images=1)


class ThinkingPhaseTests(unittest.TestCase):
    def test_natural_close_continues_with_remaining_tokens(self):
        self.assertEqual(server.after_think(eos_reason='stop_token', closed=True, think_tokens=100, max_tokens=1000, budget=None),
                         ('answer', '</think>\n\n', 899))

    def test_budget_exhaustion_forces_close(self):
        self.assertEqual(server.think_limit(1000, 300), 300)
        self.assertEqual(server.after_think(eos_reason='max_new_tokens', closed=False, think_tokens=300, max_tokens=1000, budget=300),
                         ('answer', '\n</think>\n\n', 699))

    def test_unbudgeted_exhaustion_is_length(self):
        self.assertEqual(server.think_limit(1000, None), 1000)
        self.assertEqual(server.after_think(eos_reason='max_new_tokens', closed=False, think_tokens=1000, max_tokens=1000, budget=None),
                         ('finish', 'length'))

    def test_eos_inside_thinking_finishes_without_answer(self):
        self.assertEqual(server.after_think(eos_reason='stop_token', closed=False, think_tokens=5, max_tokens=1000, budget=None),
                         ('finish', 'stop'))

    def test_no_room_left_for_answer_is_length(self):
        self.assertEqual(server.after_think(eos_reason='stop_token', closed=True, think_tokens=999, max_tokens=1000, budget=None),
                         ('finish', 'length'))
        self.assertEqual((server.finish_reason('max_new_tokens'), server.finish_reason('end_filter')), ('length', 'stop'))


class RuntimeCommandTests(unittest.TestCase):
    def command(self, **env):
        with mock.patch.dict(os.environ, env, clear=True), mock.patch('importlib.util.find_spec', return_value=object()):
            runtime = VLMRuntime('http://127.0.0.1:8011/v1', 'local-qwen')
            try:
                return runtime._server_command(8011)
            finally:
                runtime.close()

    def test_default_batch_uses_larger_cache_without_draft(self):
        command, _ = self.command()
        arg = lambda name: command[command.index(name) + 1]
        self.assertEqual((arg('--max-batch-size'), arg('--cache-tokens'), arg('--draft')), ('5', '73728', ''))
        self.assertEqual(arg('--max-model-len'), '32768')

    def test_service_is_loopback_and_batch_matches_client_slots(self):
        command, env = self.command(VLM_MAX_NUM_SEQS='2', PYTHONPATH='/somewhere')
        arg = lambda name: command[command.index(name) + 1]
        self.assertEqual((arg('--host'), arg('--port'), arg('--served-model-name')), ('127.0.0.1', '8011', 'local-qwen'))
        self.assertEqual(arg('--max-batch-size'), '2')
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['EXL3_QC_STAGING'], '1')
        server.build_parser().parse_args(command[2:])   # 服务端能解析 runtime 生成的全部参数

    def test_draft_can_be_disabled(self):
        command, _ = self.command(VLM_DRAFT_PATH='none')
        self.assertEqual(command[command.index('--draft') + 1], '')

    def test_missing_engine_rejected(self):
        runtime = VLMRuntime('http://127.0.0.1:8011/v1', 'local-qwen')
        try:
            with mock.patch('importlib.util.find_spec', return_value=None), self.assertRaisesRegex(ValueError, 'ExLlamaV3'):
                runtime._server_command(8011)
        finally:
            runtime.close()

    def test_missing_model_rejected(self):
        with self.assertRaisesRegex(ValueError, '目标模型配置不存在'):
            self.command(VLM_MODEL_PATH='/nonexistent/model')

    def test_server_refuses_non_loopback_host(self):
        command, _ = self.command()
        argv = command[2:]
        argv[argv.index('--host') + 1] = '0.0.0.0'
        with self.assertRaisesRegex(SystemExit, '回环'):
            server.main(argv)


if __name__ == '__main__':
    unittest.main()
