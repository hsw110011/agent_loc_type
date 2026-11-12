"""VLMClient 的 image_labels：标签与图片交替排列；不启动服务、不加载模型。"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agent.vlm.client import VLMClient, VLMConfig, VLMError


class ImageLabelTests(unittest.TestCase):
    def setUp(self):
        self.client = VLMClient(VLMConfig())
        self.client.runtime.ensure_started = lambda: None
        self.payloads = []
        reply = {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
        self.client._request = lambda path, payload=None: (self.payloads.append(payload), reply)[1]
        self.tmp = tempfile.TemporaryDirectory()
        self.images = []
        for k in range(2):
            path = Path(self.tmp.name)/f'{k}.png'
            Image.new('RGB', (4, 4), (k*100, 0, 0)).save(path)
            self.images.append(str(path))

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def content(self):
        return self.payloads[-1]['messages'][1]['content']

    def test_labels_interleave_before_each_image(self):
        self.client.infer('主提示', self.images, image_labels=['第 1 张：A', '第 2 张：B'])
        self.assertEqual([c['type'] for c in self.content()], ['text', 'image_url', 'text', 'image_url', 'text'])
        self.assertEqual([c['text'] for c in self.content() if c['type'] == 'text'], ['第 1 张：A', '第 2 张：B', '主提示'])

    def test_without_labels_keeps_images_then_prompt(self):
        self.client.infer('主提示', self.images)
        self.assertEqual([c['type'] for c in self.content()], ['image_url', 'image_url', 'text'])

    def test_thinking_passthrough_and_reasoning(self):
        reply = {'choices': [{'message': {'content': '{}', 'reasoning_content': '先看图'}, 'finish_reason': 'stop'}]}
        self.client._request = lambda path, payload=None: (self.payloads.append(payload), reply)[1]
        result = self.client.infer('主提示', self.images, max_tokens=2560, enable_thinking=True,
                                   thinking_token_budget=1024, reasoning_effort='medium')
        payload = self.payloads[-1]
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': True, 'reasoning_effort': 'medium'})
        self.assertEqual(payload['thinking_token_budget'], 1024)
        self.assertEqual(result.reasoning, '先看图')
        del reply['choices'][0]['message']['reasoning_content']
        self.assertEqual(self.client.infer('主提示').reasoning, '')

    def test_unbudgeted_thinking_sends_no_budget(self):
        self.client.infer('主提示', self.images, max_tokens=2560, enable_thinking=True)
        self.assertEqual(self.payloads[-1]['chat_template_kwargs'], {'enable_thinking': True})
        self.assertNotIn('thinking_token_budget', self.payloads[-1])

    def test_length_error_keeps_generated_text(self):
        reply = {'choices': [{'message': {'content': '{"a"', 'reasoning_content': '想到一半'}, 'finish_reason': 'length'}]}
        self.client._request = lambda path, payload=None: reply
        with self.assertRaises(VLMError) as caught:
            self.client.infer('主提示')
        self.assertEqual((caught.exception.raw_text, caught.exception.reasoning), ('{"a"', '想到一半'))

    def test_invalid_labels_rejected_before_request(self):
        for labels in (['只有一个'], 'A', ['A', '  '], ['A', 3]):
            with self.assertRaises(ValueError):
                self.client.infer('主提示', self.images, image_labels=labels)
        self.assertEqual(self.payloads, [])


if __name__ == '__main__':
    unittest.main()
