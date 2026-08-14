"""
Root conftest.py — required so pytest adds the project root to sys.path
(the test suite imports common.py, metrics.py, etc. directly; they're plain
modules, not a package) and so no test ever blocks on a GUI event loop.
"""

import matplotlib

matplotlib.use("Agg")  # must happen before any pyplot import, including in the modules under test

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _no_plt_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
