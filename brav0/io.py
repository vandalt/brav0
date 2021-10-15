"""
Functions to read and write ZP calibration data
"""
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from astropy.table import Table
from pandas.core.frame import DataFrame

import brav0.utils as ut


def source_tables(
    pattern: Union[str, Path],
    in_ext: str = ".rdb",
    file_col: str = "RVFILE",
    row_col: str = "ROW",
) -> DataFrame:

    flist = ut.generate_flist(pattern, ext=in_ext)

    # Load bunch of files to put in directory
    data_dict = dict()
    for fpath in flist:
        fbase = os.path.basename(fpath)
        ftype = os.path.splitext(fbase)[-1][1:]
        data_dict[fbase] = Table.read(fpath, format=ftype).to_pandas()

    return pd.concat(data_dict, names=[file_col, row_col])


def load_df(
    path: Union[str, Path], sort_col: Optional[str] = None
) -> DataFrame:
    data = pd.read_csv(path, index_col=[0, 1])
    if sort_col is not None:
        data = data.sort_values(sort_col)
    return data


def save_df(data, path: Union[str, Path], force: bool = False) -> None:
    path = Path(path)

    if path.exists() and not force:
        raise FileExistsError(
            f"File {path} exists. Use force=True to overwrite"
        )

    data.to_csv(path)


def load_zp(zpfile: Union[str, Path]) -> DataFrame:
    return pd.read_csv(zpfile)
