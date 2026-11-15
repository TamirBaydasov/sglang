# E (branch E) CPU tests for the chat encode pool's memo core.
#
# Self-contained on purpose: the memo module has no sglang imports, so these
# tests import it by file path and exercise the render/encode contracts with
# a character-level fake tokenizer (encode is per-character, so
# encode(prefix) + encode(tail) == encode(all) exactly -- any assembly-math
# bug shows up as an id mismatch, not as tolerated skew).

import importlib.util
import os
import threading
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "chat_encode_memo",
    os.path.join(
        _HERE, "..", "..", "..", "python", "sglang", "srt", "managers",
        "chat_encode_memo.py",
    ),
)
chat_encode_memo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chat_encode_memo)

ChatEncodeMemo = chat_encode_memo.ChatEncodeMemo
MIN_PREFIX_CHARS = chat_encode_memo.MIN_PREFIX_CHARS
encode_chat_text = chat_encode_memo.encode_chat_text

DOC = ("doc paragraph. " * 600)[:MIN_PREFIX_CHARS + 500]  # > MIN_PREFIX_CHARS


class CharTokenizer:
    """A stand-in HF-style tokenizer: renders and encodes per character."""

    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            tools=None, return_dict=False, **kw):
        parts = "".join(
            f"<{m.get('role', 'user')}>{m.get('content', '')}</{m.get('role', 'user')}>"
            for m in messages
        )
        return parts + ("<GEN>" if add_generation_prompt else "")

    def encode(self, text, **kw):
        return [ord(c) % 251 for c in text]

    def decode(self, ids):
        return ""


class RustyTokenizer(CharTokenizer):
    """Adds a Rust-backed fast tokenizer object for the raw-encode path."""

    class _Rust:
        def encode(self, text, **kw):
            return [ord(c) % 251 for c in text]

    def __init__(self):
        self._tokenizer = self._Rust()


def _messages(doc, question):
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": doc},
        {"role": "user", "content": question},
    ]


class TestEncodeChatText(unittest.TestCase):
    def test_wrapper_path(self):
        tok = CharTokenizer()
        self.assertEqual(encode_chat_text(tok, "abc", None, False), [97, 98, 99])

    def test_raw_path_matches_wrapper(self):
        tok = RustyTokenizer()
        self.assertEqual(
            encode_chat_text(tok, "abc", None, True),
            encode_chat_text(tok, "abc", None, False),
        )

    def test_raw_falls_back_without_rust_object(self):
        tok = CharTokenizer()
        self.assertEqual(encode_chat_text(tok, "abc", None, True), [97, 98, 99])

    def test_encode_kwargs_forwarded(self):
        tok = CharTokenizer()
        calls = []
        original = tok.encode

        def guarded(text, **kw):
            calls.append(kw)
            return original(text)

        tok.encode = guarded
        encode_chat_text(tok, "abc", {"add_special_tokens": False}, False)
        self.assertEqual(calls, [{"add_special_tokens": False}])


class TestExactLRU(unittest.TestCase):
    def test_store_get_roundtrip(self):
        memo = ChatEncodeMemo()
        memo.store_full("k", "prompt", [1, 2, 3])
        self.assertEqual(memo.get_full("k"), ("prompt", [1, 2, 3]))
        self.assertIsNone(memo.get_full("missing"))

    def test_lru_eviction(self):
        memo = ChatEncodeMemo(full_lru_size=2)
        memo.store_full("a", "pa", [1])
        memo.store_full("b", "pb", [2])
        memo.get_full("a")  # touch a
        memo.store_full("c", "pc", [3])  # evicts b
        self.assertIsNone(memo.get_full("b"))
        self.assertIsNotNone(memo.get_full("a"))
        self.assertIsNotNone(memo.get_full("c"))


class TestFamilyAssembly(unittest.TestCase):
    def test_commit_registers_family_and_assembly_reconstructs(self):
        """A sibling extends the family's anchor document (same shape key, the
        anchor grows): the assembled render and ids equal the full render."""
        tok = CharTokenizer()
        memo = ChatEncodeMemo()
        messages = _messages(DOC, "q1")
        rendered = tok.apply_chat_template(messages, tokenize=False)
        pos = rendered.find(DOC)
        self.assertGreaterEqual(pos, 0)
        ids, _ = memo.commit_full("shape", DOC, rendered, b"ek", tok.encode)
        self.assertEqual(ids, tok.encode(rendered))

        longer_doc = DOC + " and the doc continues"
        sibling = _messages(longer_doc, "q1")
        rendered2 = tok.apply_chat_template(sibling, tokenize=False)
        assembled = memo.begin_assembly("shape", longer_doc)
        self.assertIsNotNone(assembled)
        prefix, cut, postlude, version = assembled
        self.assertEqual(cut, len(DOC))
        rebuilt = prefix + longer_doc[cut:] + postlude
        self.assertEqual(rebuilt, rendered2)
        prefix_ids = memo.prefix_ids(version, b"ek", lambda: tok.encode(prefix))
        tail_ids = tok.encode(rebuilt[len(prefix):])
        self.assertEqual(prefix_ids + tail_ids, tok.encode(rendered2))

    def test_prefix_ids_encoded_once_per_config(self):
        tok = CharTokenizer()
        memo = ChatEncodeMemo()
        memo.commit_full("shape", DOC, tok.apply_chat_template(
            _messages(DOC, "q1"), tokenize=False), b"ek", tok.encode)
        prefix, _, _, version = memo.begin_assembly("shape", DOC)
        calls = []

        def counting():
            calls.append(1)
            return tok.encode(prefix)

        memo.prefix_ids(version, b"ek", counting)
        memo.prefix_ids(version, b"ek", counting)
        self.assertEqual(len(calls), 1)
        memo.prefix_ids(version, b"ek2", counting)  # other config -> new encode
        self.assertEqual(len(calls), 2)

    def test_begin_declines_unknown_family_or_short_content(self):
        memo = ChatEncodeMemo()
        self.assertIsNone(memo.begin_assembly("nope", DOC))
        tok = CharTokenizer()
        memo.commit_full("shape", DOC, tok.apply_chat_template(
            _messages(DOC, "q1"), tokenize=False), b"ek", tok.encode)
        self.assertIsNone(memo.begin_assembly("shape", "short"))
        self.assertIsNone(memo.begin_assembly("shape", "x" * len(DOC)))

    def test_commit_without_anchor_in_render_registers_nothing(self):
        memo = ChatEncodeMemo()
        ids, _ = memo.commit_full("shape", DOC, "<render-without-anchor>", b"ek",
                                  CharTokenizer().encode)
        self.assertEqual(ids, CharTokenizer().encode("<render-without-anchor>"))
        self.assertIsNone(memo.begin_assembly("shape", DOC))

    def test_family_lru_eviction(self):
        tok = CharTokenizer()
        memo = ChatEncodeMemo()
        docs = {}
        for i in range(70):  # > _FAMILY_LRU_SIZE (64)
            doc = f"{i}." + DOC[len(str(i)) + 1:]
            docs[i] = doc
            memo.commit_full(f"shape{i}", doc, tok.apply_chat_template(
                _messages(doc, "q"), tokenize=False), b"ek", tok.encode)
        self.assertIsNone(memo.begin_assembly("shape0", docs[0]))
        self.assertIsNotNone(memo.begin_assembly("shape69", docs[69]))


class TestBootstrapSerialization(unittest.TestCase):
    def test_claim_is_exclusive_and_wait_releases_after_commit(self):
        tok = CharTokenizer()
        memo = ChatEncodeMemo()
        self.assertTrue(memo.claim_creation("shape"))
        self.assertFalse(memo.claim_creation("shape"))
        self.assertFalse(memo.wait_family("shape", 0.05))  # still creating

        results = []

        def waiter():
            results.append(memo.wait_family("shape", 5.0))

        t = threading.Thread(target=waiter)
        t.start()
        memo.commit_full("shape", DOC, tok.apply_chat_template(
            _messages(DOC, "q1"), tokenize=False), b"ek", tok.encode)
        t.join(5.0)
        self.assertEqual(results, [True])
        self.assertIsNotNone(memo.begin_assembly("shape", DOC))

    def test_commit_without_family_registration_releases_waiters(self):
        memo = ChatEncodeMemo()
        self.assertTrue(memo.claim_creation("shape"))
        memo.commit_full("shape", DOC, "<no-anchor>", b"ek", CharTokenizer().encode)
        # Event was set; a fresh claim is possible again.
        self.assertTrue(memo.claim_creation("shape"))

    def test_wait_after_creator_finished_is_true(self):
        tok = CharTokenizer()
        memo = ChatEncodeMemo()
        memo.commit_full("shape", DOC, tok.apply_chat_template(
            _messages(DOC, "q1"), tokenize=False), b"ek", tok.encode)
        self.assertTrue(memo.wait_family("shape", 0.0))


if __name__ == "__main__":
    unittest.main()
