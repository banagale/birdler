from pathlib import Path
import postprocess as pp

def test_atempo_cmd(monkeypatch, tmp_path):
    calls = []
    def fake_run(cmd, check=True):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"WAV")
    monkeypatch.setattr(pp.subprocess, 'run', fake_run)
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'WAV')
    pp.change_tempo_in_place(wav, 0.9)
    assert any('atempo=0.9' in ' '.join(c) for c in calls)
