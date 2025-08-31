import importlib


def test_import_path_exposes_main():
    m = importlib.import_module("birdler")
    assert hasattr(m, "main")

