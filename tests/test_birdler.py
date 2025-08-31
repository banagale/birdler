import pytest
from pathlib import Path

import birdler


class DummyArgs:
    def __init__(self, text=None, text_file=None):
        self.text = text
        self.text_file = text_file
        self.max_chars = 280
        self.hard_max_chars = 360


def test_default_chunks():
    # Default behavior returns at least one chunk of DEFAULT_TEXT
    args = DummyArgs()
    chunks = birdler.get_text_chunks(args)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_direct_text_short():
    text = "Hello world"
    args = DummyArgs(text=text)
    chunks = birdler.get_text_chunks(args)
    assert chunks == [text]


def test_direct_text_long_splits():
    # Create a script significantly longer than default max chunk length
    long_text = "x" * 600
    args = DummyArgs(text=long_text)
    chunks = birdler.get_text_chunks(args)
    # Should split into multiple chunks and reconstruct
    assert len(chunks) >= 2
    assert "".join(chunks) == long_text


def test_text_file(tmp_path):
    content = "File based script"
    file = tmp_path / "script.txt"
    file.write_text(content)
    args = DummyArgs(text_file=file)
    chunks = birdler.get_text_chunks(args)
    assert chunks == [content]
