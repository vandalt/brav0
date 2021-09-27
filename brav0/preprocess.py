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


# =============================================================================
# Define helper functions
# =============================================================================
def filter_odo(files: Series, odos: Union[Series, list[str]]) -> Series:
    return ~files.str[:7].isin(odos)


# All bad ID filtering methods
ID_FILTER_METHODS = {
    "odo": filter_odo,
}


def get_clip_mask(x: Series, nsig: float) -> np.ndarray:
    return ~sigma_clip(x.values, sigma=nsig).mask


def keep_self_template(data: DataFrame, file_col: str = "RVFILE") -> DataFrame:
    file_series = pd.Series(data.index.get_level_values(file_col).unique())
    keep_files = file_series[
        file_series.str[:-4].str.split("_").agg(lambda x: x[1] == x[2])
    ]
    return data.loc[keep_files]


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
    try:
        data = requests.get(url)
    except requests.exceptions.RequestException:
        warnings.warn("Could not load bad id list, returning an empty list")
        return []

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


# =============================================================================
# Data Filtering functions
# =============================================================================
def filter_nan_values(data: DataFrame, used_cols: Optional[list[str]] = None):
    """
    Filter NaNs in columns that are used for futher calculations

        :param data: Dataframe with full dataset
        :type data: DataFrame
        :param used_cols: Columns to check, None checks all, defaults to None
        :type used_cols: Optional[list[str]], optional
    """
    return data.dropna(subset=used_cols).copy()


def sort_dataset(data: DataFrame, time_col: str):
    """
    Filter NaNs in columns that are used for futher calculations

        :param data: Dataframe with full dataset
        :type data: DataFrame
        :param time_col: Time column label
        :type time_col: str
    """
    return data.sort_values(time_col)


def filter_bad_ids(
    data: DataFrame,
    bad_ids: Union[Series, list[str], str],
    filter_col: str,
    filter_method: Union[str, Callable] = "odo",
):
    """
    Filter observation IDs that are flagged as badded, either from an
    online Google Sheet or from a Series or list.

    :param data: Dataset
    :type data: DataFrame
    :param bad_ids: Bad IDs collection or url
    :type bad_ids: Union[Series, list[str], str]
    :param filter_col: Column to use to filter IDs
    :type filter_col: str
    :param filter_method: Filtering methods to apply on filter_col. This is
                          either the name of a known method
                          (see preprocess.FILTER_METHOD) or a function taking
                          the ID column and a bad IDs as input. Must return a
                          boolean series.
                          Defaults to "odo"
    :type filter_method: Union[str, Callable], optional
    :raises ValueError: If filter_method is not an known method or a function
    :raises TypeError: If the filter_method does not return a boolean series.
    """
    if isinstance(bad_ids, str):
        if bad_ids.startswith("https://"):
            bad_ids = get_bad_id_list(bad_ids)
        else:
            msg = (
                "Only https links are supported for the bad ID sheet."
                " Make sure your string starts with 'https://'."
                " If this is a single bad ID string, put it in a list."
            )
            raise ValueError(msg)

    if filter_method in ID_FILTER_METHODS:
        keep_mask = ID_FILTER_METHODS[filter_method](data[filter_col], bad_ids)
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

    print(f"{np.sum(~keep_mask)} bad IDs filtered")
    return data[keep_mask].copy()


def filter_sig_clip(
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

    print(f"{np.sum(~clipped_mask)} points removed from sigma clipping")
    return data[clipped_mask].copy()


def filter_snr_cut(
    data: DataFrame, snr_col: str, snr_goal_col: str, snr_frac: int = 0.7
):
    mask = (data[snr_col] / data[snr_goal_col]) >= snr_frac

    print(f"{np.sum(~mask)} points removed from SNR cut")

    return data[mask].copy()


def filter_equant(
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

    print(f"{np.sum(~mask)} points removed from error quantile cut")

    return data[mask]


# -----------------------------------------------------------------------------
# Combined cleanup function
# -----------------------------------------------------------------------------
PP_CLEAN_FUNCTIONS = {
    "nan": filter_nan_values,
    "sigma": filter_sig_clip,
    "ID": filter_bad_ids,
    "SNR": filter_snr_cut,
    "error": filter_equant,
}


def cleanup(
    data: DataFrame,
    plist: list[str] = None,
    bad_id_url: Optional[str] = None,
    id_filter_col: Optional[str] = None,
    clip_col: Optional[str] = None,
    nsig_clip: float = 3.0,
    group_col: str = Optional[None],
    equant_col: Optional[str] = None,
    equant_cut: float = 0.95,
    snr_col: Optional[str] = None,
    time_col: Optional[str] = None,
    snr_goal_col: Optional[str] = None,
    snr_frac: float = 0.7,
    used_cols: Optional[list[str]] = None,
):

    # TODO: A lot of the checks here should be in their respective mask func
    ilength = len(data)
    if "nan" in plist:
        data = filter_nan_values(data, used_cols)

    if "sort" in plist:
        if time_col is None:
            raise ValueError("time_col is required to sort the data")
        data = sort_dataset(data, time_col)

    if "ID" in plist:
        data = filter_bad_ids(data, bad_id_url, id_filter_col)

    if "sigma" in plist:
        data = filter_sig_clip(
            data,
            clip_col,
            nsig=nsig_clip,
            group_name=group_col,
        )

    if "SNR" in plist:
        if snr_col is None or snr_goal_col is None:
            warnings.warn(
                "snr_col and snr_goal_col are needed to run SNR cut. Skipping."
            )
        else:
            data = filter_snr_cut(data, snr_col, snr_goal_col, snr_frac)

    if "equant" in plist:
        if (
            equant_cut is not None
            and equant_col is not None
            and (0.0 > equant_cut or equant_cut > 1.0)
        ):
            raise ValueError("err_quant_cut must be between 0 and 1")
        elif equant_cut is None or equant_col is None:
            warnings.warn(
                "snr_col and snr_goal_col are needed to run SNR cut. Skipping."
            )
        else:
            data = filter_equant(
                data,
                equant_col,
                equant_cut,
                group_name=group_col,
            )
    flength = len(data)

    print(f"Total {ilength-flength} (non-binned) points removed by cleanup.")

    return data


# =============================================================================
# Extra functions to format dataframe and prepare analysis
# =============================================================================
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
            df = keep_self_template(df)
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
    """
    Get binned dataset

    :param data: Full dataset
    :type data: DataFrame
    :param vrad_col: Column name with RV values
    :type vrad_col: str
    :param svrad_col: Column name with RV errors
    :type svrad_col: str
    :param extra_pairs: Dictionary of extra value-error column pairs that
                        don't start with 0, defaults to None
    :type extra_pairs: Optional[dict[str, str]], optional
    """

    vrad_cols = data.columns[data.columns.str.startswith(vrad_col)].tolist()
    svrad_cols = data.columns[data.columns.str.startswith(svrad_col)].tolist()
    wmean_pairs = dict(zip(vrad_cols, svrad_cols))

    if extra_pairs is not None:
        wmean_pairs = {**wmean_pairs, **extra_pairs}

    return ut.get_binned_data(data, wmean_pairs=wmean_pairs)


def keep_obj_list(
    data: DataFrame, obj_list: list[str], obj_col: str = "OBJECT"
):
    """
    Fitler dataset to keep objects from a list

    :param data: Full dataset
    :type data: DataFrame
    :param obj_list: List of objects to keep
    :type obj_list: list[str]
    :param obj_col: Label of object column, defaults to "OBJECT"
    :type obj_col: str, optional
    """

    data = data.copy()

    if obj_list is None:
        warnings.warn(
            "obj_list is None, this will do nothing",
            category=RuntimeWarning,
        )
        return data
    elif len(obj_list) == 0:
        warnings.warn("obj_list is an empty list. This will do nothing.")
        return data
    elif isinstance(obj_list, str):
        obj_list = [obj_list]

    if obj_col in data.index.names:
        data = data.loc[obj_list]
    elif obj_col in data.columns:
        msg = f"{obj_col} is not an index." "Trying to filter with columns"
        warnings.warn(msg)
        obj_mask = data[obj_col].isin(obj_list)
        data = data[obj_mask]
    else:
        raise ValueError(f"obj_col={obj_col} is not an index or a column.")

    return data


def add_archive_name(
    data: DataFrame,
    obj_label: str,
    extra_maps: Optional[dict[str, str]] = None,
) -> DataFrame:
    """
    Add column with exoplanet archive names to dataframes.

    Replaces GLXXX and GJXXX with GJ XXX.

    :param data: Initial dataframe
    :type data: DataFrame
    :param obj_label: Label of object index
    :type obj_label: str
    :param extra_maps: Extra mappings from object name to archive name
                       (ovevrrides default subsitutions), defaults to None
    :type extra_maps: Optional[Dict]
    :return: Dataframe with an ARCHIVE column
    :rtype: DataFrame
    """
    data = data.copy()

    # Exoplanet archive has all GL000 and GJ000 with GJ 000
    obj_names = data.index.get_level_values(obj_label)
    obj_mask = ~obj_names.isin(list(extra_maps))
    data["ARCHIVE"] = obj_names.copy()
    data.loc[obj_mask, "ARCHIVE"] = obj_names[obj_mask].str.replace(
        "^GJ|GL", "GJ ", regex=True
    )

    # If we have user-provided values, they replace the string manips above
    if extra_maps is not None:
        data.ARCHIVE = data.ARCHIVE.replace(extra_maps)

    return data
