import glob

import brav0.utils as ut
import pytest


@pytest.fixture
def glob_output(data_glob):
    return glob.glob(str(data_glob))


def test_pathglob_str(glob_output, data_glob):
    flist = ut.pathglob(str(data_glob))
    assert flist == glob_output


def test_pathglob_path(glob_output, data_glob):
    flist = ut.pathglob(data_glob)
    assert flist == glob_output
