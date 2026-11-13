"""Exercise real tokenizer IDs and the serving method without importing NPU code."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
ROOT = Path(__file__).resolve().parents[1]
TREES = [ROOT / "sglang/tb-pr-chat-encode-pool", ROOT / "tb-scratch"]


@pytest.fixture(params=TREES, ids=["pr2", "combined"])
def source(request):
    root = request.param
    if not root.is_dir():
        pytest.skip("requires the corresponding local feature worktree")
    path = root / "python/sglang/srt/managers/chat_encode_memo.py"
    spec = importlib.util.spec_from_file_location("memo_under_test", path)
    memo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memo)
    tree = ast.parse(
        (root / "python/sglang/srt/entrypoints/openai/serving_chat.py").read_text()
    )
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_memo_render_and_encode"
    )
    # Run the actual changed serving method while avoiding unrelated torch/NPU imports.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "encode_chat_text": memo.encode_chat_text,
        "envs": SimpleNamespace(
            SGLANG_CHAT_ENCODE_RAW=SimpleNamespace(get=lambda: True)
        ),
    }
    exec(compile(module, str(root), "exec"), namespace)
    return memo, namespace["_memo_render_and_encode"]


def tokenizer():
    rust = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab={"[UNK]": 0, "a": 1, "b": 2, "ab": 3, "[BOS]": 4, "[EOS]": 5},
            merges=[("a", "b")],
            unk_token="[UNK]",
        )
    )
    rust.post_processor = tokenizers.processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 4), ("[EOS]", 5)]
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=rust, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    )
    tok.chat_template = (
        "{{ messages[0]['content'] }}{% if add_generation_prompt %}b{% endif %}"
    )
    return tok


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"add_special_tokens": False},
        {"add_special_tokens": True},
        {"truncation": True, "max_length": 4},
    ],
)
def test_real_encoding_and_special_token_options(source, kwargs):
    memo, _ = source
    tok = tokenizer()
    assert isinstance(tok.backend_tokenizer.encode("ab"), tokenizers.Encoding)
    assert memo.encode_chat_text(tok, "ababa", kwargs, raw=True) == tok.encode(
        "ababa", **kwargs
    )


def test_custom_wrapper_is_preserved(source):
    memo, _ = source
    base = tokenizer()

    class Custom(type(base)):
        def encode(self, text, **kwargs):
            return [777] + super().encode(text, **kwargs)

    tok = Custom(tokenizer_object=base.backend_tokenizer, unk_token="[UNK]")
    assert memo.encode_chat_text(
        tok, "ab", {"add_special_tokens": False}, raw=True
    ) == [777, 3]


def test_family_changed_tail_and_repeated_requests_encode_whole_prompt(source):
    module, method = source
    tok = tokenizer()
    server = SimpleNamespace(tokenizer_manager=SimpleNamespace(tokenizer=tok))
    memo = module.ChatEncodeMemo()
    anchor = "a" * (module.MIN_PREFIX_CHARS + 1)
    # This cut changes the BPE result; the old character-only mock could not expose it.
    assert tok.encode(anchor, add_special_tokens=False) + tok.encode(
        "b", add_special_tokens=False
    ) != tok.encode(anchor + "b", add_special_tokens=False)
    for index, text in enumerate([anchor, anchor + "b", anchor + "ba"]):
        messages = [{"role": "user", "content": text}]
        kwargs = dict(
            tools=None,
            template_kwargs={},
            encode_kwargs={"add_special_tokens": False},
            cache_key=str(index),
            memo=memo,
        )
        rendered, ids = method(server, messages, **kwargs)
        assert rendered == tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert ids == tok.encode(rendered, add_special_tokens=False)
        ids.append(999)
        _, again = method(server, messages, **kwargs)
        assert again == tok.encode(rendered, add_special_tokens=False)
