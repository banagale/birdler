import json, csv
from pathlib import Path
from datetime import datetime

def save_settings(base_output_path: Path, output_files: list[str], params: dict):
    base_output_path = Path(base_output_path)
    safe = dict(params)
    for key in ('text', 'text_input'):
        if key in safe:
            safe[key] = ''
    safe['output_audio_files'] = list(map(str, output_files))
    safe['generation_time'] = datetime.now().isoformat()
    base_output_path.with_suffix('.settings.json').write_text(json.dumps(safe, indent=2))
    with base_output_path.with_suffix('.settings.csv').open('w', newline='') as f:
        w = csv.writer(f)
        for k,v in safe.items():
            w.writerow([k, v])
