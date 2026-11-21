"""Load a script of experiments/ as a module, so tests can call its functions."""

import importlib.util
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"


@pytest.fixture
def load_script():
    def load(name: str):
        spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
