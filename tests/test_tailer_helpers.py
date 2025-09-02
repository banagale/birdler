from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_tail_module():
    # Import tools.codex_session_tail after ensuring root on sys.path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("tools.codex_session_tail")


def test_split_sentences_basic():
    m = _load_tail_module()
    text = "Hello world. This is Birdler! Ok? Done"
    out = m._split_sentences(text)
    assert out[:3] == ["Hello world.", "This is Birdler!", "Ok?"]


def test_limit_for_speech_first_and_max():
    m = _load_tail_module()
    text = "One. Two is here. Three is also here."
    only_one = m.limit_for_speech(text, max_chars=100, first_sents=1)
    assert only_one == "One."
    clipped = m.limit_for_speech("A " * 200, max_chars=50, first_sents=0)
    assert len(clipped) <= 51 and clipped.endswith("…")


def test_strip_and_has_codeblocks():
    m = _load_tail_module()
    fenced = "Before\n```\ncode here\n```\nAfter"
    assert m.has_codeblock(fenced)
    assert m.strip_codeblocks(fenced) == "Before After"
    inline_long = "This has `some really really long inline code snippet that triggers` strip"
    assert m.has_codeblock(inline_long)
    indented = "    x = 1\n    y = 2\nDone"
    assert m.has_codeblock(indented)


def test_extract_headlines_and_bullets():
    m = _load_tail_module()
    txt = """
    What's new
    - Fast server path
    - Status ticker

    Notes
    * Uses cached embeddings
    """
    heads = m.extract_headlines(txt)
    assert "What’s New" in heads or "What's New" in heads
    assert "Notes" in heads
    bullets = m.extract_bullet_leads(txt, max_items=2)
    assert "Fast server path" in bullets
    assert bullets.endswith(".")


def test_canon_and_near_duplicate():
    m = _load_tail_module()
    a = m._canon(" Hello, World!  ")
    b = m._canon("hello world")
    assert a == b
    from collections import deque

    recent = deque([b], maxlen=5)
    assert m.is_near_duplicate("Hello world!!", recent, 0.9)


def test_phrase_cache_path_structure(tmp_path: Path):
    m = _load_tail_module()
    p = m.phrase_cache_path(tmp_path, voice="ripley", atempo=None, phrase="Skipping a code block here.")
    # expected: tmp/_phrases/ripley/atempo-1.0/<slug>.wav
    assert p.parts[-4] == "_phrases"
    assert p.parts[-3] == "ripley"
    assert p.parts[-2].startswith("atempo-")
    assert p.suffix == ".wav"

