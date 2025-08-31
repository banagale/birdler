import sys
from pathlib import Path

import pytest

import birdler


def test_youtube_extract_creates_output_dir(monkeypatch, tmp_path):
    # Prepare a non-existing output directory
    out_dir = tmp_path / "nonexistent"
    assert not out_dir.exists()

    # Simulate CLI args: use youtube-url path to short-circuit TTS logic
    monkeypatch.setattr(sys, 'argv', ['birdler.py', '--youtube-url', 'http://example.com',
                                      '--output-dir', str(out_dir)])
    # Ensure which() returns a dummy command and subprocess.run is a no-op
    monkeypatch.setattr(birdler, 'which', lambda name: 'dummy-cmd')
    monkeypatch.setattr(birdler.subprocess, 'run', lambda *args, **kwargs: None)

    # Run main()
    rc = birdler.main()
    # Check return code and directory creation
    assert rc == 0
    assert out_dir.exists() is True
