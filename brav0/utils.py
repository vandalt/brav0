import glob
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series


def pathglob(pattern: Union[str, Path]) -> list[str]:
    """
    Tiny glob wrapper to handle python Path object from pathlib

    :param pattern: Path or string representing a glob pattern
    :type pattern: Union[str, Path]
    :return: List of individual paths
    :rtype: list[str]
    """
    return glob.glob(str(pattern))


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
