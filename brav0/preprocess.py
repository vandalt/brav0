from typing import Callable, Optional, Union

import numpy as np
from astropy.stats import sigma_clip
from pandas.core.frame import DataFrame
from pandas.core.series import Series


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
