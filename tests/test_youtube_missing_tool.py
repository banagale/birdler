import sys
import birdler


def test_youtube_tool_missing(monkeypatch, tmp_path):
    # Ensure CLI exits with an error and message when yt-dlp/youtube-dl is unavailable
    monkeypatch.setattr(birdler, 'which', lambda *_: None)
    out_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'birdler.py', '--youtube-url', 'http://example.com', '--output-dir', str(out_dir)
    ])
    rc = birdler.main()
    assert rc == 1
