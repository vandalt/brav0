import glob
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from box import Box
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from tqdm import tqdm
from xarray.core.dataset import Dataset

from brav0.model import ZeroPointModel


def pathglob(pattern: Union[str, Path]) -> list[str]:
    """
    Tiny glob wrapper to handle python Path object from pathlib

    :param pattern: Path or string representing a glob pattern
    :type pattern: Union[str, Path]
    :return: List of individual paths
    :rtype: list[str]
    """
    return glob.glob(str(pattern))


def generate_flist(pattern: Union[str, Path], ext: str = ".rdb") -> list[str]:
    pattern = Path(pattern)
    if pattern.is_dir():
        # Glob all files in directory
        flist = pathglob(pattern / f"*{ext}")
    else:
        # Glob pattern directly
        flist = pathglob(pattern)

    return flist


def append_to_dirpath(path: Path, extra: str) -> Path:
    """
    Add a string to a path object (usually useful for directories)

    :param path: Initial path
    :type path: Path
    :param extra: Appended string
    :type extra: str
    :return: Path with the string appended to its basename
    :rtype: Path
    """
    return path.parent / (path.name + extra)


def get_wmean(data: DataFrame, col_pairs: dict[str, str]) -> Series:
    """
    Get the weighted mean and error of a dataframe using column pairs

    :param data: Input dataframe
    :type data: DataFrame
    :param col_pairs: Column pairs (value -> error mapping)
    :type col_pairs: Dict
    :return: Series with the weighted mean and the error for each binned value
    :rtype: Series
    """

    val_cols = pd.Index(list(col_pairs.keys()))
    err_cols = pd.Index(list(col_pairs.values()))
    vals = data[val_cols].astype(float)
    errs = data[err_cols].astype(float)
    errs2 = errs ** 2

    output = pd.Series(index=data.columns, dtype=float)

    # Get values and error columns where all nan errors
    nan_col_mask = errs2.isna().all().values
    val_cols_nan = val_cols[nan_col_mask]
    err_cols_nan = err_cols[nan_col_mask]
    nan_cols = val_cols_nan.union(err_cols_nan)
    output[nan_cols] = np.nan

    output[nan_cols] = np.nan

    good_cols = data.columns[~data.columns.isin(nan_cols)]
    val_cols_good = val_cols[val_cols.isin(good_cols)]
    err_cols_good = err_cols[err_cols.isin(good_cols)]
    good_vals = vals[val_cols_good].values
    good_errs2 = errs2[err_cols_good].values

    output[val_cols_good] = np.nansum(
        good_vals / good_errs2, axis=0
    ) / np.nansum(1 / good_errs2, axis=0)
    output[err_cols_good] = np.sqrt(1 / np.nansum(1 / good_errs2, axis=0))

    return output


def get_binned_data(
    data: DataFrame, wmean_pairs: Optional[dict[str, str]] = None
) -> DataFrame:
    """
    Bin dataframe with three possible operations:
    - Weighted mean: for wmean_pairs columns, takes both a value and its error
    - Regular mean: take mean of values
    - First: keep first value

    The dataframe is grouped by file and date, then binned.

    :param data: DataFrame with OBJECT and DATE-OBS info, plus other quantities
    :type data: DataFrame
    :param wmean_pairs: Pairs of value->error columns to use for weighted means
    :type wmean_pairs: Dict, optional
    :return: Dataframe binned per day
    :rtype: DataFrame
    """

    # Get a list of all columsn involved in the weighted means
    # num_cols = data.select_dtypes(include=np.number).columns
    # nonum_cols = data.select_dtypes(exclude=np.number).columns
    # data = data.astype
    grouped_data = data.groupby(["OBJECT", "DATE-OBS"])
    if wmean_pairs is not None:
        wmean_all_cols = list(wmean_pairs.keys())
        wmean_all_cols.extend(list(wmean_pairs.values()))

        # For other columns, either use a simple mean for numbers or keep first
        no_wmean_cols = data.columns[~data.columns.isin(wmean_all_cols)]
        mean_cols = (
            data[no_wmean_cols].select_dtypes(include=np.number).columns
        )
        first_cols = (
            data[no_wmean_cols].select_dtypes(exclude=np.number).columns
        )

        # Dictionary to tell agg which function to use on which columns
        agg_funcs = {
            **dict.fromkeys(mean_cols, "mean"),
            **dict.fromkeys(first_cols, "first"),
        }

        # Perform operations
        agg_data = grouped_data.agg(agg_funcs)
        binned_data = grouped_data.apply(get_wmean, wmean_pairs)
        binned_data[agg_data.columns] = agg_data

        # We now reset the ROW index that was lost in the binning
        # This makes the data more uniform with the pre-binning data
        no_date_data = binned_data.reset_index("DATE-OBS", drop=True)
        rows_per_file = no_date_data.groupby("OBJECT").size().apply(np.arange)
        rows = np.concatenate(rows_per_file)
        row_ind = pd.Index(rows, name="ROW")
        binned_data = no_date_data.set_index(row_ind, append=True)
    else:
        mean_cols = data.select_dtypes(include=np.number).columns
        first_cols = data.select_dtypes(exclude=np.number).columns

        # Dictionary to tell agg which function to use on which columns
        agg_funcs = {
            **dict.fromkeys(mean_cols, "mean"),
            **dict.fromkeys(first_cols, "first"),
        }

        binned_data = grouped_data.agg(agg_funcs)

    return binned_data


def bin_tbl(
    tbl: Table,
    wmean_pairs: dict[str, str],
) -> Table:

    tbl2_dict = {colname: [] for colname in tbl.colnames}

    dates = tbl["DATE-OBS"]
    udates = np.unique(dates)

    for i in tqdm(range(len(udates))):
        epoch = udates[i]
        epoch_mask = dates == epoch

        itbl = tbl[epoch_mask]

        for colname in tbl.colnames:

            if colname in wmean_pairs:
                # get value and error for this udate
                vals = itbl[colname]
                errs = itbl[wmean_pairs[colname]]
                # get error^2
                errs2 = errs ** 2
                # deal with all nans
                if np.sum(np.isfinite(errs2)) == 0:
                    value = np.nan
                    err_value = np.nan
                else:
                    # get 1/error^2
                    value = np.nansum(vals / errs2) / np.nansum(1 / errs2)
                    err_value = np.sqrt(1 / np.nansum(1 / errs2))
                # push into table
                tbl2_dict[colname].append(value)
                tbl2_dict[wmean_pairs[colname]].append(err_value)
            # -----------------------------------------------------------------
            # if no weighted mean indication, try to mean the column or if not
            #   just take the first value
            elif colname not in wmean_pairs.values():
                # try to produce the mean of rdb table
                # noinspection PyBroadException
                try:
                    tbl2_dict[colname].append(np.mean(itbl[colname]))
                except TypeError:
                    tbl2_dict[colname].append(itbl[colname][0])
    tbl2 = Table()
    for colname in tbl2_dict:
        tbl2[colname] = tbl2_dict[colname]

    return tbl2


def get_obj_vals(
    data: DataFrame, obj_col: str = "OBJECT", unique: bool = False
):

    if obj_col in data.index.names:
        ovals = data.index.get_level_values(obj_col).values
    elif obj_col in data.columns:
        msg = f"{obj_col} is not an index." "Trying to filter with columns"
        warnings.warn(msg)
        ovals = data[obj_col].values
    else:
        raise ValueError(f"obj_col={obj_col} is not an index or a column.")

    return ovals.values if not unique else np.unique(ovals)


def tt_atleast_1d(x):
    """
    Attempt of theano equiavalent to numpy atleast_1d
    """
    if x.broadcastable == ():
        return x.dimshuffle("x")
    return x


def load_config(path: Union[str, Path]):
    with open(path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    return Box(config)


def make_unique_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path_ini = path
    while path.exists():
        # This should not block more than one second unless dir already
        # exist for some reason. Still probably safer/clearer than _0, _1, etc.
        ext_time = "_" + datetime.now().strftime("%y%m%d_%H%M%S")
        path = append_to_dirpath(path_ini, ext_time)

    path.mkdir(parents=True)

    return path


def save_map_dict(map_dict: dict, path: Union[Path, str], force: bool = False):
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"File {path} exists. Use force=True to overwrite"
        )
    pkl_ext = [".pkl", ".pickle"]
    if path.suffix not in pkl_ext:
        raise ValueError(
            "Please use one of the following file extension for a pickle"
            f" file: {pkl_ext}"
        )

    with open(path, "wb") as pfile:
        pickle.dump(map_dict, pfile)


def get_config_params(config: Box):
    model_params = config.model_parameters
    if isinstance(model_params, str):
        if not model_params.endswith(".yml"):
            raise ValueError(
                "model_params should be a dictionary or a yml file"
            )
        params_file = Path(model_params)
        # If not an absolut path, assume from config dir
        if not params_file.is_absolute():
            parent_dir = Path(config.config).parent
            params_file = parent_dir / params_file
        model_params = load_config(params_file)
        if "model_parameters" in model_params:
            model_params = model_params["model_parameters"]
        return model_params
    elif isinstance(model_params, dict):
        return model_params
    else:
        raise TypeError(
            "model_parameters should be a dictionary or a path to a file"
        )


def get_substr_keys(
    substr: str,
    model: Optional[ZeroPointModel] = None,
    post: Optional[Dataset] = None,
    map_dict: Optional[dict] = None,
) -> list[str]:
    if model is not None:
        keys = [k for k in model.named_vars.keys() if substr in k]
    elif post is not None:
        keys = [k for k in post.data_vars.keys() if substr in k]
    elif map_dict is not None:
        keys = [k for k in map_dict.keys() if substr in k]
    else:
        raise TypeError("One of model or post is required.")

    return keys


def print_data_info(
    data: DataFrame, config: Box, wn_dict: Optional[dict[str, float]] = None
):
    for obj in data.index.get_level_values("OBJECT").unique():
        odata = data.loc[obj]
        print(f"Info for {obj}")
        print(f"  Mean RV error: {np.mean(odata[config.svrad_col])}")
        print(f"  Median RV error: {np.median(odata[config.svrad_col])}")
        print(f"  RV scatter: {np.std(odata[config.svrad_col])}")
        if wn_dict is not None:
            print(f"  White noise term: {wn_dict[obj]}")


def get_summary(group):

    d = dict()
    d["npts"] = group.shape[0]
    d["range"] = group.rjd.max() - group.rjd.min()
    d["vrad_std"] = group.vrad.std()
    d["svrad_mean"] = group.svrad.mean()

    return pd.Series(d, index=list(d))
