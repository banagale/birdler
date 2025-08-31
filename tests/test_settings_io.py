from pathlib import Path
from settings_io import save_settings
import json, csv

def test_save_settings_artifacts(tmp_path):
    base = tmp_path / 'out.wav'
    base.write_bytes(b'WAV')
    params = {'seed': 123, 'text_input': 'Hello'}
    save_settings(base, [base], params)
    js = base.with_suffix('.settings.json')
    cs = base.with_suffix('.settings.csv')
    assert js.exists() and cs.exists()
    data = json.loads(js.read_text())
    assert data.get('seed') == 123
    assert data.get('text_input') == ''
