# E2a (branch E): the chat render+encode memo.
#
# Serves `_render_and_encode_chat_template` (serving_chat). Two layers:
#
# 1. An exact-repeat LRU keyed by the full request fingerprint (template,
#    messages, tools, kwargs, add_generation_prompt): a repeat returns the
#    stored ids with no render and no encode.
# 2. A family prefix assembly keyed by the family shape (template, roles,
#    every NON-anchor message, tools, kwargs): requests whose long anchor
#    document differs but whose shape is the same reuse the family's rendered
#    prefix and re-encode only the tail. The first request of a family renders
#    in full and registers the family; concurrent first requests are
#    serialized (one claims the creation, the rest wait, then assemble).
#
# The prefix split is character-exact by construction (the anchor content is
# located verbatim inside the rendered prompt), but the ENCODE split at the
# cut can skew by a few tokens at a BPE boundary -- the ids come from
# `encode(prefix) + encode(tail)` instead of one `encode(all)`. This is
# documented, accepted skew: prompt ids feed the model, not a byte contract,
# and the family reuse exists to kill the 1M-token GIL-wave encode burst, not
# to reproduce bit-identical tokenizations. Requests that need exact ids take
# the full-render path (exact LRU or fallback).
#
# Host-side only (branch E); nothing here touches device state.

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

# E2a: a message must carry at least this many characters to be the family
# anchor. Below it the prefix-encode saving cannot pay for the machinery.
MIN_PREFIX_CHARS = 4096

_FULL_LRU_SIZE = 256
_FAMILY_LRU_SIZE = 64


def encode_chat_text(
    tokenizer,
    text: str,
    encode_kwargs: Optional[Dict] = None,
    raw: bool = False,
) -> List[int]:
    """Encode chat prompt text, optionally through the raw Rust tokenizer.

    `raw` bypasses the HF wrapper and calls the underlying fast tokenizer's
    Rust object directly (`tokenizer._tokenizer.encode`). The wrapper adds
    Python-level pre/post-processing around the same GIL-released Rust call,
    so the raw path saves wrapper overhead on very long prompts. Silently
    falls back to the wrapper when the tokenizer has no Rust object (slow
    tokenizers, fake tokenizers in tests).
    """
    kwargs = dict(encode_kwargs or {})
    if raw:
        rust = getattr(tokenizer, "_tokenizer", None)
        if rust is not None and hasattr(rust, "encode"):
            return list(rust.encode(text, **kwargs))
    return list(tokenizer.encode(text, **kwargs))


class _Family:
    """One registered family: a rendered prefix anchored at a long message."""

    __slots__ = (
        "anchor",
        "cut",
        "rendered_prefix",
        "postlude",
        "version",
        "prefix_ids",
    )

    def __init__(
        self,
        anchor: str,
        rendered_prefix: str,
        postlude: str,
    ):
        self.anchor = anchor
        # How many anchor characters the stored prefix covers (everything the
        # first render had of the anchor -- the whole anchor).
        self.cut = len(anchor)
        self.rendered_prefix = rendered_prefix
        self.postlude = postlude
        self.version = 1
        # (encode_kwargs_key -> ids): the prefix encode, computed once per
        # encode configuration and shared by every later sibling.
        self.prefix_ids: Dict[bytes, List[int]] = {}


class ChatEncodeMemo:
    """Exact-repeat LRU + family prefix assembly for chat render+encode."""

    def __init__(self, full_lru_size: int = _FULL_LRU_SIZE):
        self._lock = threading.Lock()
        # cache_key -> (rendered_prompt, prompt_ids)
        self._full: OrderedDict = OrderedDict()
        self._full_lru_size = full_lru_size
        # shape_key -> _Family
        self._families: "OrderedDict[bytes, _Family]" = OrderedDict()
        # shape_key -> Event set while the claiming thread renders
        self._creating: Dict[bytes, threading.Event] = {}
        # version -> (_Family, {encode_kwargs_key: ids})
        self._by_version: Dict[int, tuple] = {}
        self._next_version = 1

    # ---- exact repeats ----

    def get_full(self, cache_key) -> Optional[Tuple[str, List[int]]]:
        with self._lock:
            hit = self._full.get(cache_key)
            if hit is not None:
                self._full.move_to_end(cache_key)
            return hit

    def store_full(self, cache_key, rendered_prompt: str, prompt_ids: List[int]) -> None:
        with self._lock:
            self._full[cache_key] = (rendered_prompt, list(prompt_ids))
            self._full.move_to_end(cache_key)
            while len(self._full) > self._full_lru_size:
                self._full.popitem(last=False)

    # ---- family bootstrap serialization ----

    def claim_creation(self, shape_key) -> bool:
        """Try to become the family's creator. True: render in full and call
        commit_full. False: someone else is rendering it -- wait_family."""
        with self._lock:
            if shape_key in self._creating:
                return False
            self._creating[shape_key] = threading.Event()
            return True

    def wait_family(self, shape_key, timeout: float) -> bool:
        """Wait for the claiming thread to commit. True: retry
        begin_assembly. False: the creator stalled or timed out -- render in
        full rather than block the request past the bootstrap timeout.

        Never pops the claim: only the creator's commit_full removes it. A
        timed-out waiter that dropped the claim would let a second thread
        claim creation and render the same family again. A creator that dies
        before commit leaves the claim set -- later waiters time out and fall
        back to full renders (correct, just not amortized).
        """
        with self._lock:
            event = self._creating.get(shape_key)
        if event is None:
            # No claim outstanding: either it already finished (family is in)
            # or there was never a creator for this key.
            return shape_key in self._families
        event.wait(timeout)
        return shape_key in self._families

    # ---- family assembly ----

    def begin_assembly(
        self, shape_key, content: str
    ) -> Optional[Tuple[str, int, str, int]]:
        """The (prefix, cut, postlude, version) to assemble `content` from.

        None when the family is unknown, still being created, or the content
        does not extend the family anchor far enough for the reuse to pay.
        """
        with self._lock:
            family = self._families.get(shape_key)
            if family is None:
                return None
            if len(content) < family.cut:
                return None
            if content[: family.cut] != family.anchor[: family.cut]:
                return None
            self._families.move_to_end(shape_key)
            return family.rendered_prefix, family.cut, family.postlude, family.version

    def prefix_ids(
        self,
        version: int,
        encode_kwargs_key,
        encode_fn: Callable[[], List[int]],
    ) -> Optional[List[int]]:
        """The family prefix's ids, encoded once per encode configuration.

        encode_fn runs on the caller's thread (a chat-encode pool worker);
        the memo only guarantees the encode happens once.
        """
        with self._lock:
            entry = self._by_version.get(version)
            if entry is None:
                return None
            family, id_cache = entry
            cached = id_cache.get(encode_kwargs_key)
            if cached is not None:
                return list(cached)
        # Encode outside the lock: it can take seconds on a 1M-char prefix,
        # and holding the memo lock through it would serialize every family.
        ids = list(encode_fn())
        with self._lock:
            entry = self._by_version.get(version)
            if entry is None:
                return None
            family, id_cache = entry
            if family.version == version:
                id_cache[encode_kwargs_key] = ids
            return list(ids)

    def commit_full(
        self,
        shape_key,
        content: str,
        rendered_prompt: str,
        encode_kwargs_key,
        encode: Callable[[str], List[int]],
    ) -> Tuple[List[int], None]:
        """Register the family from a full render; return the full ids.

        The anchor content must appear verbatim in the render (it does for
        text messages under every stock chat template); when it does not, the
        family is simply not registered and future requests fall through to
        full renders -- never a wrong prompt.
        """
        prompt_ids = list(encode(rendered_prompt))
        pos = rendered_prompt.find(content)
        if pos >= 0 and len(content) >= MIN_PREFIX_CHARS:
            family = _Family(
                anchor=content,
                rendered_prefix=rendered_prompt[: pos + len(content)],
                postlude=rendered_prompt[pos + len(content) :],
            )
            version = family.version
            with self._lock:
                version = self._next_version
                self._next_version += 1
                family.version = version
                self._families[shape_key] = family
                self._by_version[version] = (family, {})
                self._families.move_to_end(shape_key)
                while len(self._families) > _FAMILY_LRU_SIZE:
                    evicted = self._families.popitem(last=False)
                    self._by_version.pop(evicted[1].version, None)
                # This claim's waiters may proceed (to assemble, or to a
                # fresh full render if they do not match the family).
                event = self._creating.pop(shape_key, None)
            if event is not None:
                event.set()
        else:
            with self._lock:
                event = self._creating.pop(shape_key, None)
            if event is not None:
                event.set()
        return prompt_ids, None
