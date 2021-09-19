from pathlib import Path

import pytest


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def data_glob(data_dir):
    return data_dir / "*.rdb"
