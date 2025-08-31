from progress import log_progress

def test_progress_prints(capsys):
    log_progress(1, 4)
    out = capsys.readouterr().out
    assert '[PROGRESS] 1/4' in out
