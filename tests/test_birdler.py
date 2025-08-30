import pytest
from pathlib import Path

import birdler


class DummyArgs:
    def __init__(self, text=None, text_file=None):
        self.text = text
        self.text_file = text_file


def test_default_chunks():
    # Default behavior returns the hardcoded Big Bird chunks
    args = DummyArgs()
    chunks = birdler.get_text_chunks(args)
    assert isinstance(chunks, list)
    assert len(chunks) == 3


def test_direct_text_short():
    text = "Hello world"
    args = DummyArgs(text=text)
    chunks = birdler.get_text_chunks(args)
    assert chunks == [text]


def test_direct_text_long_splits():
    # Create a script longer than default max chunk length
    default_chunks = birdler.get_text_chunks(DummyArgs())
    max_def = max(len(c) for c in default_chunks)
    long_text = "x" * (max_def + 50)
    args = DummyArgs(text=long_text)
    chunks = birdler.get_text_chunks(args)
    # Should split into same number of default chunks and reconstruct
    assert len(chunks) == len(default_chunks)
    assert "".join(chunks) == long_text


def test_text_file(tmp_path):
    content = "File based script"
    file = tmp_path / "script.txt"
    file.write_text(content)
    args = DummyArgs(text_file=file)
    chunks = birdler.get_text_chunks(args)
    assert chunks == [content]
