from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def data_glob(data_dir):
    return data_dir / "*.txt"


@pytest.fixture()
def round_equal():
    def _round_equal(a, b, tol):
        return np.all(np.round(a, tol) == np.round(b, tol))

    return _round_equal


@pytest.fixture
def tol():
    return 7


@pytest.fixture
def simple_values():
    return np.array([1.0, 2, 3, 4, 5])


@pytest.fixture
def simple_weights():
    return np.array([0.2, 0.1, 0.6, 0.01, 0.15])
