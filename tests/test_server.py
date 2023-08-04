import unittest
import requests
from urllib import parse


class EndToEndServerTest(unittest.TestCase):
    first_prompt = 'Alan Turing is'
    second_prompt = 'Albert Einstein'

    def setUp(self) -> None:

        return super().setUp()
    
    def test_with_prompt(self):
        resp = self._send_prompt("Alan Turing is", 10)
        self.assertEqual(200, resp.status_code)
        self.assertEqual("a British mathematician and computer scientist who was", resp.text.strip())

    def test_with_another_prompt(self):
        resp = self._send_prompt("Albert Einstein", 8)
        self.assertEqual(200, resp.status_code)
        self.assertEqual("was a German-born theoretical physicist", resp.text.strip())

    def _send_prompt(self, prompt, max_tokens):
        query = parse.urlencode([("prompt", prompt), ("max-tokens", max_tokens)])
        url = f'http://localhost:5000/generate?{query}'
        return requests.get(url)