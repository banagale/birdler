import types
import birdler

def test_naming_includes_gen_and_seed(monkeypatch):
    class A: pass
    args = A()
    args.voice = 'v'
    args.audio_sample = None
    args.text = None
    args.text_file = None
    args.run_index = 2
    args.seed = 123
    chunks = ['hello world']
    # Freeze time
    monkeypatch.setattr(birdler, 'time', types.SimpleNamespace(time=lambda: 111))
    name = birdler._derive_outname(args, chunks)
    assert '_gen2' in name and '_seed123' in name and name.endswith('.wav')
