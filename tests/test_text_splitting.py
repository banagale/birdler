import pytest
from pathlib import Path

import birdler


class DummyArgs:
    def __init__(self, text=None, text_file=None):
        self.text = text
        self.text_file = text_file
        self.max_chars = 280
        self.hard_max_chars = 360


def test_text_file_long_splits(tmp_path, capsys):
    # create a long script well over the default max chunk length
    long_content = 'x' * 600
    script_path = tmp_path / 'long.txt'
    script_path.write_text(long_content)
    args = DummyArgs(text_file=script_path)
    chunks = birdler.get_text_chunks(args)
    assert len(chunks) >= 2
    # concatenation should reconstruct original content
    assert ''.join(chunks) == long_content


def test_direct_text_long_splits(capsys):
    long_text = 'y' * 600
    args = DummyArgs(text=long_text)
    chunks = birdler.get_text_chunks(args)
    assert len(chunks) >= 2
    assert ''.join(chunks) == long_text
