"""
Functions to read and write ZP calibration data
"""
import os
from typing import Union
from pathlib import Path

import pandas as pd
from astropy.table import Table
from pandas.core.frame import DataFrame

import brav0.utils as ut


def source_tables(
    pattern: Union[str, Path],
    in_ext=".rdb",
    file_col: str = "RVFILE",
    row_col: str = "ROW",
) -> DataFrame:

    pattern = Path(pattern)
    if pattern.is_dir():
        # Glob all files in directory
        flist = ut.pathglob(pattern / f"*{in_ext}")
    else:
        # Glob pattern directly
        flist = ut.pathglob(pattern)

        # Load bunch of files to put in directory
        data_dict = dict()
        for fpath in flist:
            fbase = os.path.basename(fpath)
            ftype = os.path.splitext(fbase)[-1][1:]
            data_dict[fbase] = Table.read(fpath, format=ftype).to_pandas()

    return pd.concat(data_dict, names=[file_col, row_col])


def load_df(path: Union[str, Path]) -> DataFrame:
    return pd.read_csv(path, index_col=[0, 1])
