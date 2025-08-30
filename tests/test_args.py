import sys
from pathlib import Path
import pytest

import birdler


def test_parse_args_default(monkeypatch):
    # no text flags => both None
    monkeypatch.setattr(sys, 'argv', ['birdler.py'])
    args = birdler.parse_args()
    assert args.text is None
    assert args.text_file is None


def test_parse_args_accepts_text_file(tmp_path, monkeypatch):
    script = tmp_path / 'script.txt'
    script.write_text('hello')
    monkeypatch.setattr(sys, 'argv', ['birdler.py', '--text-file', str(script)])
    args = birdler.parse_args()
    assert args.text_file == script
    assert args.text is None


def test_parse_args_accepts_text(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['birdler.py', '--text', 'direct text'])
    args = birdler.parse_args()
    assert args.text == 'direct text'
    assert args.text_file is None


@pytest.mark.parametrize('args_list', [
    ['birdler.py', '--text', 'a', '--text-file', 'b.txt'],
    ['birdler.py', '--text-file', 'b.txt', '--text', 'a'],
])
def test_mutually_exclusive_text_args(monkeypatch, args_list):
    monkeypatch.setattr(sys, 'argv', args_list)
    with pytest.raises(SystemExit):
        birdler.parse_args()
