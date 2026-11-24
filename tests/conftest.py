"""Shared fixtures. All tests run on synthetic data (see synthetic_data.py)."""
from __future__ import annotations

import os
import sys

import pytest

TESTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(TESTS, "..")))
sys.path.insert(0, TESTS)

from synthetic_data import make_dataset  # noqa: E402


def _quiet(*args, **kwargs):
    pass


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """data/-like folder with converted/ (3 labeled + 3 unlabeled sausages)."""
    base = tmp_path_factory.mktemp("data")
    make_dataset(str(base / "converted"), n_sausages=3, days=2)
    return base


@pytest.fixture(scope="session")
def trained(synthetic_dir):
    """synthetic_dir plus models/ trained with tiny settings."""
    from ml_core.training import load_windows, train_and_evaluate
    windows = load_windows(str(synthetic_dir / "converted"), log=_quiet)
    train_and_evaluate(windows, str(synthetic_dir / "models"), quick=True, log=_quiet)
    return synthetic_dir
