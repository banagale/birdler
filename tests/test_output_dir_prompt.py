import sys
from pathlib import Path

import pytest

import birdler


@pytest.mark.parametrize("user_input, should_create, expected_rc", [
    ("y", True, 0),
    ("n", False, 1),
])
def test_output_dir_prompt(monkeypatch, tmp_path, user_input, should_create, expected_rc):
    # Prepare a non-existing output directory
    out_dir = tmp_path / "nonexistent"
    assert not out_dir.exists()

    # Simulate CLI args: use youtube-url path to short-circuit TTS logic
    monkeypatch.setattr(sys, 'argv', ['birdler.py', '--youtube-url', 'http://example.com',
                                      '--output-dir', str(out_dir)])
    # Patch input() to simulate user response
    monkeypatch.setattr('builtins.input', lambda prompt: user_input)
    # Ensure which() returns a dummy command and subprocess.run is a no-op
    monkeypatch.setattr(birdler, 'which', lambda name: 'dummy-cmd')
    monkeypatch.setattr(birdler.subprocess, 'run', lambda *args, **kwargs: None)

    # Run main()
    rc = birdler.main()
    # Check return code
    assert rc == expected_rc
    # Directory should exist only if user_input was affirmative
    assert out_dir.exists() == should_create
