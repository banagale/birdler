from pathlib import Path
import postprocess as pp

def test_denoise_calls_cli_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(pp, 'has_cli', lambda name: True)
    def fake_run(cmd, check=False):
        # ffmpeg writes intermediate; denoise writes output
        if cmd[0] == 'ffmpeg':
            Path(cmd[-1]).write_bytes(b'WAV')
            class R: returncode = 0
            return R()
        if cmd[0] == 'denoise':
            Path(cmd[-1]).write_bytes(b'WAV')
            class R: returncode = 0
            return R()
    monkeypatch.setattr(pp.subprocess, 'run', fake_run)
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'WAV')
    ok = pp.denoise_in_place(wav)
    assert ok
