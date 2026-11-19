"""Regression coverage for chat memo rendering contracts, without server imports."""

import ast
import importlib.util
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import jinja2
import orjson

ROOT = Path(__file__).resolve().parents[2]


def serving():
    path = ROOT / "python/sglang/srt/managers/chat_encode_memo.py"
    spec = importlib.util.spec_from_file_location("memo_contract", path)
    memo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memo)
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    tree = ast.parse(path.read_text())
    methods = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_render_and_encode_chat_template", "_memo_render_and_encode")
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = dict(
        orjson=orjson,
        encode_chat_text=memo.encode_chat_text,
        envs=SimpleNamespace(SGLANG_CHAT_ENCODE_RAW=SimpleNamespace(get=lambda: False)),
        parallel_prompt_encode=lambda tokenizer, text, kwargs: tokenizer.encode(
            text, **kwargs
        ),
        render_chat_segments=lambda *args, **kwargs: ("custom segments",),
        encode_prompt_segments=lambda *args, **kwargs: [321],
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *methods], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    cls = type(
        "Serving",
        (),
        {
            name: namespace[name]
            for name in ("_render_and_encode_chat_template", "_memo_render_and_encode")
        },
    )
    server = cls()
    tokenizer = Tokenizer()
    server.tokenizer_manager = SimpleNamespace(
        tokenizer=tokenizer, chat_encode_memo=memo.ChatEncodeMemo()
    )
    server.template_manager = SimpleNamespace(python_chat_template=None)
    server._prompt_text_round_trip_is_lossy = False
    server._chat_template_cache = OrderedDict()
    return server


class Tokenizer:
    chat_template = "{% for key in tools[0] %}{{key}} {% endfor %}"

    def __init__(self):
        self.encodes = 0

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = jinja2.Template(self.chat_template).render(messages=messages, **kwargs)
        return [999] if tokenize else text

    def encode(self, text, **kwargs):
        self.encodes += 1
        return list(text.encode())

    def decode(self, ids):
        return "unused"


def render(server, tools):
    return server._render_and_encode_chat_template(
        [{"role": "user", "content": "test"}],
        tools=[tools],
        template_kwargs={},
        encode_kwargs={},
        use_cache=False,
    )[0]


class TestChatMemoContracts(unittest.TestCase):
    def test_dictionary_order_is_part_of_cache_identity(self):
        server = serving()
        self.assertEqual(render(server, {"a": 1, "b": 2}), list(b"a b "))
        self.assertEqual(render(server, {"b": 2, "a": 1}), list(b"b a "))
        self.assertEqual(render(server, {"a": 1, "b": 2}), list(b"a b "))
        self.assertEqual(server.tokenizer_manager.tokenizer.encodes, 2)

    def test_lossy_tokenizer_keeps_direct_template_tokens(self):
        server = serving()
        server._prompt_text_round_trip_is_lossy = True
        self.assertEqual(render(server, {"a": 1}), [999])
        self.assertEqual(server.tokenizer_manager.tokenizer.encodes, 0)

    def test_python_template_keeps_segment_encoder(self):
        server = serving()
        server.template_manager.python_chat_template = object()
        self.assertEqual(render(server, {"a": 1}), [321])
        self.assertEqual(server.tokenizer_manager.tokenizer.encodes, 0)


if __name__ == "__main__":
    unittest.main()
