from pathlib import Path
import postprocess as pp

def test_normalize_ffmpeg_cmd(monkeypatch, tmp_path):
    calls = []
    def fake_run(cmd, check=True):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"WAV")
    monkeypatch.setattr(pp.subprocess, 'run', fake_run)
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'WAV')
    pp.normalize_ffmpeg_in_place(wav, method='ebu', I=-24, TP=-2, LRA=7)
    assert any('ffmpeg' in c[0] for c in calls)
