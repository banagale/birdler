import pytest
from pathlib import Path

import birdler


class DummyArgs:
    def __init__(self, text=None, text_file=None):
        self.text = text
        self.text_file = text_file


def test_text_file_long_splits(tmp_path, capsys):
    # create a long script just over the max default chunk length
    default_chunks = birdler.get_text_chunks(DummyArgs())
    max_def = max(len(c) for c in default_chunks)
    long_content = 'x' * (max_def + 10)
    script_path = tmp_path / 'long.txt'
    script_path.write_text(long_content)
    args = DummyArgs(text_file=script_path)
    chunks = birdler.get_text_chunks(args)
    out = capsys.readouterr().out
    assert f"Script length {len(long_content)} > {max_def}" in out
    assert len(chunks) == len(default_chunks)
    # concatenation should reconstruct original content
    assert ''.join(chunks) == long_content


def test_direct_text_long_splits(capsys):
    default_chunks = birdler.get_text_chunks(DummyArgs())
    max_def = max(len(c) for c in default_chunks)
    long_text = 'y' * (max_def + 5)
    args = DummyArgs(text=long_text)
    chunks = birdler.get_text_chunks(args)
    out = capsys.readouterr().out
    assert f"Script length {len(long_text)} > {max_def}" in out
    assert len(chunks) == len(default_chunks)
    assert ''.join(chunks) == long_text
