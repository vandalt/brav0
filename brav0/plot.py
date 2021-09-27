from pathlib import Path
from typing import Optional, Union

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
from arviz.data.inference_data import InferenceData
from astropy.timeseries import LombScargle
from pandas.core.frame import DataFrame


def plot_all(data: DataFrame, dist: bool = True, alpha=1.0, ax=None):
    if ax is None:
        ax = plt

    csize = 2
    if not dist:
        ax.errorbar(
            data.rjd,
            data.vrad,
            data.svrad,
            fmt="k.",
            capsize=csize,
            alpha=alpha,
        )
    else:
        for obj in data.index.get_level_values("OBJECT").unique():
            dat_obj = data.loc[obj]
            ax.errorbar(
                dat_obj.rjd,
                dat_obj.vrad,
                dat_obj.svrad,
                fmt=".",
                capsize=csize,
                label=obj,
                alpha=alpha,
            )


def plot_pred(tpred, pred, pred_std, ax=None, color="r", label=None):
    if ax is None:
        ax = plt
    # plt.errorbar(model.t, model.vrad, yerr=model.svrad, fmt="k.", capsize=2)
    ax.plot(tpred, pred, zorder=100, color=color, label=label)
    art = ax.fill_between(
        tpred,
        pred - pred_std,
        pred + pred_std,
        alpha=0.5,
        zorder=100,
        color=color,
    )
    art.set_edgecolor("none")


def plot_trace(
    trace: InferenceData,
    var_names: Optional[list[str]] = None,
    filter_vars: Optional[str] = None,
    savepath: Optional[str] = None,
):
    az.plot_trace(trace, var_names=var_names, filter_vars=filter_vars)
    if savepath is not None:
        plt.savefig(savepath)


def plot_corner(
    trace: InferenceData,
    var_names: Optional[list[str]] = None,
    filter_vars: Optional[str] = None,
    savepath: Optional[str] = None,
    **corner_kwargs,
):
    corner.corner(
        trace, var_names=var_names, filter_vars=filter_vars, **corner_kwargs
    )
    if savepath is not None:
        plt.savefig(savepath)


def plot_all_objects(
    data: DataFrame,
    ocol: str = "OBJECT",
    tcol: str = "rjd",
    rvcol: str = "vrad",
    ervcol: str = "svrad",
    out_dir: Union[str, Path] = None,
    orientation: str = "horizontal",
    show: bool = False,
):

    objects = list(data.index.get_level_values(ocol).unique())
    for obj in objects:
        odata = data.loc[obj]

        # Setup figure
        if orientation == "horizontal":
            order = (1, 2)
        elif orientation == "vertical":
            order = (2, 1)
        else:
            raise ValueError("Orientation should be vertical or horizontal.")

        fig, (axrv, axper) = plt.subplots(*order, figsize=(8, 8))

        freq, pwr = plot_series_periodogram(
            odata[tcol].values,
            odata[rvcol].values,
            odata[ervcol].values,
            axes=(axrv, axper),
        )

        # Finalize plot
        plt.suptitle(obj)
        plt.tight_layout()
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            plt.savefig(out_dir / f"{obj}_rv_perio.pdf")
            np.savetxt(
                out_dir / f"{obj}_rv_perio.txt", np.array([freq, pwr]).T
            )
        if show:
            plt.show()
        plt.close(fig)


def plot_series_periodogram(
    t: np.ndarray,
    rv: np.ndarray,
    erv: np.ndarray,
    axes: Optional[tuple[plt.Axes]] = None,
    freq: Optional[np.ndarray] = None,
    fap_val: float = 0.001,
    title: Optional[Union[str, list[str]]] = None,
    orientation: str = "horizontal",
) -> tuple[np.ndarray, np.ndarray]:
    """

    :param t: Time values
    :type t: astropy.table.column.Column
    :param rv: RV values
    :type rv: astropy.table.column.Column
    :param erv: RV errors
    :type erv: astropy.table.column.Column
    :param axes: Axes (RV and periodogram)
    :type axes: matplotlib.axes._subplots.AxesSubplot
    :param freq: Frequency array, autogenerated if None (Default: None)
    :type freq: Union[None, astropy.units.quantity.Quantity]
    :param fap_val: FAP
    :type fap_val: float
    :param title: Title of the axes
    :type title: Optional[Union[str, List[str]]]
    :return:
    :rtype: Tuple[astropy.units.quantity.Quantity,
    astropy.units.quantity.Quantity]
    """
    # Assign axes
    if axes is None:
        if orientation == "horizontal":
            _, axes = plt.subplots(1, 2)
        elif orientation == "vertical":
            _, axes = plt.subplots(2, 1)
        else:
            raise ValueError("Orientation should be vertical or horizontal.")
    axr, axp = axes

    if isinstance(title, str):
        title = [
            title,
        ] * 2

    # Calculate LS periodogram and window function
    ls = LombScargle(t, rv, erv)
    if freq is None:
        freq, pwr = ls.autopower()
    else:
        pwr = ls.power(freq)
    fap = ls.false_alarm_level(fap_val)

    wf_ls = LombScargle(t, np.ones_like(t), center_data=False, fit_mean=False)
    wf_freq = freq.copy()
    wf_pwr = wf_ls.power(wf_freq)

    # Plot RV
    axr.errorbar(
        t,
        rv - np.median(rv),
        yerr=erv,
        fmt="k.",
        capsize=2,
    )
    # axr.plot(t, np.ones_like(tbl['rjd']))
    axr.set_ylabel("RV - Median [m/s]")
    axr.set_xlabel("RJD")
    if title is not None:
        axr.set_title(title[0])

    # Plot periodotitles[i]
    axp.plot(1 / freq, pwr, "k", label="RV periodogram")
    axp.plot(1 / wf_freq, wf_pwr, "r", label="Window function")
    axp.axhline(
        fap, linestyle="--", color="k", label=f"{100 * (1 - fap_val)}% FAP"
    )
    # axp.axvline(2.64, linestyle="-", color="b", zorder=-1,
    #             alpha=0.6)
    axp.set_xscale("log")
    axp.set_ylabel("Power")
    axp.set_xlabel("Period [d]")
    if title is not None:
        axp.set_title(title[1])
    axp.legend()
    return freq, pwr
