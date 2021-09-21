import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import requests
from astropy.stats import sigma_clip
from astropy.table import Table
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import brav0.utils as ut


def filter_odo(files: Series, odos: Union[Series, list[str]]) -> Series:
    return ~files.str[:7].isin(odos)


FILTER_METHODS = {
    "odo": filter_odo,
}


def mask_bad_ids(
    data: DataFrame,
    bad_ids: Union[Series, list[str]],
    filter_col: str,
    filter_method: Union[str, Callable] = "odo",
):
    if filter_method in FILTER_METHODS:
        keep_mask = FILTER_METHODS[filter_method](data[filter_col], bad_ids)
    else:
        try:
            keep_mask = filter_method(data[filter_col], bad_ids)
        except TypeError:
            raise ValueError(
                "filter_method must be either a string for know FILTER_METHODS"
                " or a function that takes data series and bad IDs series."
            )
        if not isinstance(keep_mask, Series) or keep_mask.dtype != "bool":
            raise TypeError("filter_method must return a pandas bool series")

    return keep_mask


def get_clip_mask(x: Series, nsig: float) -> np.ndarray:
    return ~sigma_clip(x.values, sigma=nsig).mask


def mask_sig_clip(
    data: DataFrame,
    clip_col: str,
    nsig: float = 3.0,
    group_name: Optional[str] = None,
) -> Series:
    """
    Perform sigma clipping on a pandas dataframe using a single column.

    :param data: Input dataframe
    :type data: DataFrame
    :param clip_col: Label of column used for clipping
    :type clip_col: str
    :param nsig: Number of sigmas to clip from, defaults to 3
    :type nsig: int, optional
    :param group_name: Name of the index/column used to group the dataframe.
                       The clipping is done per group.  Useful to group per
                       object or run, defaults to None
    :type group_name: Optional[str], optional
    :return: Dataset after sigma clipping of outliers
    :rtype: DataFrame
    """

    series = data[clip_col]
    if group_name is not None:
        clipped_mask = series.groupby(group_name).transform(
            get_clip_mask, nsig
        )
    else:
        clipped_mask = get_clip_mask(series, nsig)

    return clipped_mask


def mask_snr_cut(
    data: DataFrame, snr_col: str, snr_goal_col: str, snr_frac: int = 0.7
):
    return (data[snr_col] / data[snr_goal_col]) >= snr_frac


def mask_quant(
    data: DataFrame,
    qcol: str,
    quant: float = 0.95,
    group_name: Optional[str] = None,
) -> DataFrame:
    """
    Filter dataframe based to keep only values below a given quantile of a
    column.

    :param data: Input dataframe
    :type data: DataFrame
    :param qcol: Column to use for quantile calculation
    :type qcol: str
    :param quant: Quantile to use, defaults to 0.95
    :type quant: float, optional
    :param group_name: Name of the index/column used to group the dataframe,
                       useful to group per object or run before filtering,
                       defaults to None
    :type group_name: Optional[str], optional
    :return: Dataset filtered with quantile
    :rtype: DataFrame
    """

    series = data[qcol]
    if group_name is not None:
        qvals = series.groupby(group_name).quantile(quant)
        quant_align, data_align = qvals.align(data)
        assert data_align.equals(data)
        mask = series <= quant_align
    else:
        qval = series.quantile(quant)
        mask = series <= qval

    return mask


def get_bad_id_list(
    url: str, id_col: str = "ODOMETER", check_cols: Optional[list[str]] = None
) -> list[str]:
    """
    Get list of bad observation IDs/odometers from a google sheet.

    One column must be the observation IDs. Other columsn are expected to have
    "TRUE" or "FALSE" values with any column name.

    :param url: URL of bad odometer sheet
    :type url: str
    :return: pandas serries with bad odo strings
    :rtype: pd.Series
    """
    # fetch data
    data = requests.get(url)
    df = Table.read(data.text, format="ascii").to_pandas()

    if check_cols is None:
        check_cols = df.columns.drop(id_col)
    elif not np.all([cc in df.columns for cc in check_cols]):
        raise ValueError("check_cols must be Google sheet columns")

    # Booleans are read as text with requests + astropy/pandas -> convert
    df[id_col] = df[id_col].astype(str)  # ensure strings
    df[check_cols] = df[check_cols] == "TRUE"

    # Keep only values marked in check_cols
    mask = df[check_cols].any(axis=1)
    df = df[mask]

    return df[id_col].to_list()


def keep_self_mask(data: DataFrame, file_col: str = "RVFILE") -> DataFrame:
    file_series = pd.Series(data.index.get_level_values(file_col).unique())
    keep_files = file_series[
        file_series.str[:-4].str.split("_").agg(lambda x: x[1] == x[2])
    ]
    return data.loc[keep_files]


def index_with_obj(
    df: DataFrame,
    file_obj_ind: int,
    file_obj_sep: str = "_",
    file_col: str = "RVFILE",
    obj_col: str = "OBJECT",
    row_col: str = "ROW",
    force_self_mask: bool = False,
) -> DataFrame:
    """
    Add object column to dataframe based on raw RV file names. This puts a
    strong constrain on input filenames that could be removed in the
    future.

    This also checks that each object is there only once: we don't expect
    two RV timeseries for the same object in our calib data (e.g. with
    different masks or templates).


    :param file_obj_ind: Index of the split filename where the object name
                         is
    :type file_obj_ind: int
    :param file_obj_sep: Separator between filename parts, defaults to "_"
    :type file_obj_sep: str, optional
    """

    df = df.copy()

    # NOTE: this is a strong assumption on raw format/filenames
    #       can think of ways  to improve it in the future
    if obj_col in df.columns:
        msg = f"There is already an {obj_col} column, doing nothing."
        warnings.warn(msg, category=RuntimeWarning)
        return

    # Get filenames in pandas index object and split/index to get obj names
    file_lvl = df.index.get_level_values(file_col)
    objects = (
        file_lvl.str.split(file_obj_sep).str[file_obj_ind].set_names(obj_col)
    )

    # Add object column and reset index
    df[obj_col] = objects

    if (df.reset_index().groupby(obj_col).nunique()[file_col] > 1).any():
        if force_self_mask:
            df = keep_self_mask(df)
        else:
            msg = "Dataframe should not have more than one file per object"
            raise ValueError(msg)

    df = df.reset_index()
    df = df.set_index([obj_col, row_col])
    # Move file column at the end
    df[file_col] = df.pop(file_col)

    return df


def bin_dataset(
    data: DataFrame,
    vrad_col: str,
    svrad_col: str,
    extra_pairs: Optional[dict[str, str]] = None,
):

    vrad_cols = data.columns[data.columns.str.startswith(vrad_col)].tolist()
    svrad_cols = data.columns[data.columns.str.startswith(svrad_col)].tolist()
    wmean_pairs = dict(zip(vrad_cols, svrad_cols))

    if extra_pairs is not None:
        wmean_pairs = {**wmean_pairs, **extra_pairs}

    return ut.get_binned_data(data, wmean_pairs=wmean_pairs)


def keep_obj_list(
    data: DataFrame, obj_list: list[str], obj_col: str = "OBJECT"
):

    if obj_list is None:
        warnings.warn(
            "obj_list is None, this will do nothing",
            category=RuntimeWarning,
        )
        return
    elif len(obj_list) == 0:
        warnings.warn("obj_list is an empty list. This will do nothing.")
        return
    elif isinstance(obj_list, str):
        obj_list = [obj_list]

    if obj_col in data.index.names:
        data = data.loc[obj_list]
    elif obj_col in data.columns:

        obj_mask = data[obj_col].isin(obj_list)

        data = data[obj_mask]
    else:
        msg = f"{obj_col} is not an index." "Trying to filter with columns"
        warnings.warn(msg)

    return data


def preprocess(
    data: DataFrame,
    bad_id_url: Optional[str] = None,
    id_filter_col: Optional[str] = None,
    clip_col: Optional[str] = None,
    nsig_clip: float = 3.0,
    group_col: str = Optional[None],
    equant_col: Optional[str] = None,
    err_quant_cut: float = 0.95,
    snr_col: Optional[str] = None,
    snr_goal_col: Optional[str] = None,
    snr_frac: float = 0.7,
    used_cols: Optional[list[str]] = None,
):

    # TODO: A lot of the checks here should be in their respective mask func

    data = data.dropna(subset=used_cols)

    # -------------------------------------------------------------------------
    # Remove bad odometers and do sigma clipping for outliers
    # -------------------------------------------------------------------------
    if bad_id_url is not None:
        bad_id_list = get_bad_id_list(bad_id_url)
        mask = mask_bad_ids(data, bad_id_list, id_filter_col)
        data = data[mask]
    if nsig_clip is not None and nsig_clip > 0.0:
        mask = mask_sig_clip(
            data,
            clip_col,
            nsig=nsig_clip,
            group_name=group_col,
        )
        data = data[mask]

    # -------------------------------------------------------------------------
    # If required parameters are passed, do snr cut
    # -------------------------------------------------------------------------
    if (
        snr_col is not None
        and snr_goal_col is not None
        and snr_frac is not None
    ):
        if not 0.0 <= snr_frac <= 1.0:
            raise ValueError("snr_frac must be between 0 and 1")
        do_snr_cut = True
    # if some but not all are set, raise error
    elif (
        snr_col is not None or snr_goal_col is not None or snr_frac is not None
    ):
        msg = "All or none of snr_col, snr_goal_col, snr_frac must be set"
        raise ValueError(msg)
    else:
        do_snr_cut = False

    if do_snr_cut:
        mask = mask_snr_cut(data, snr_col, snr_goal_col, snr_frac)
        data = data[mask]

    # -------------------------------------------------------------------------
    # If required parameters are passed, do error quantile clipping
    # -------------------------------------------------------------------------
    # TODO: simplify logic
    if (
        err_quant_cut is not None
        and 0.0 <= err_quant_cut < 1.0
        and equant_col is not None
    ) or equant_col is None:
        pass
    else:
        raise ValueError("err_quant_cut must be between 0 and 1")

    if equant_col is not None:
        mask = mask_quant(
            data,
            equant_col,
            err_quant_cut,
            group_name=group_col,
        )
        data = data[mask]

    return data
