from typing import Optional

import arviz as az
import corner
import matplotlib.pyplot as plt
from arviz.data.inference_data import InferenceData
from pandas.core.frame import DataFrame

from brav0.model import GPModel


def plot_all(data: DataFrame, dist: bool = True, alpha=1.0):
    csize = 2
    if not dist:
        plt.errorbar(
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
            plt.errorbar(
                dat_obj.rjd,
                dat_obj.vrad,
                dat_obj.svrad,
                fmt=".",
                capsize=csize,
                label=obj,
                alpha=alpha,
            )


def plot_pred(model: GPModel, pred, pred_std):
    # plt.errorbar(model.t, model.vrad, yerr=model.svrad, fmt="k.", capsize=2)
    plt.plot(model.tpred, pred)
    plt.fill_between(model.tpred, pred - pred_std, pred + pred_std, alpha=0.5)


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
        trace,
        var_names=var_names,
        filter_vars=filter_vars,
        **corner_kwargs
    )
    if savepath is not None:
        plt.savefig(savepath)
