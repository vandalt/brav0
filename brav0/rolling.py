from typing import Optional, Union

import aesara_theano_fallback.tensor as tt
import numpy as np
from pymc3.theanof import take_along_axis
from theano.tensor.var import TensorConstant, TensorVariable

import brav0.utils as ut
from brav0.lexsort_op import lexsort


def wmed(
    values: Union[np.ndarray, TensorVariable, TensorConstant],
    weights: Union[np.ndarray, TensorVariable, TensorConstant],
    use_np: bool = True,
    axis: Optional[int] = None,
) -> Union[float, TensorVariable, TensorConstant]:
    """
    Calculate a weighted median with numpy or theano.

    :param values: Sample values to get the median.
    :type values: Union[np.ndarray, TensorVariable, TensorConstant]
    :param weights: Weights associated to each value.
    :type weights: Union[np.ndarray, TensorVariable, TensorConstant]
    :return: Weighted median of the sample values.
    :rtype: Union[float, TensorVariable, TensorConstant]
    """
    if axis not in [None, 1]:
        raise ValueError("Only None and 1 are currently supported for axis")
    diff_ax = axis if axis is not None else 0
    if use_np:
        values = np.array(values)
        weights = np.array(weights)
        diff_ax = axis if axis is not None else 0
        if not np.all(np.diff(values, axis=diff_ax) >= 0):
            # lexsorts will sort weights after values
            # if value==0 and weight!=0, avoids having weird results
            inds = np.lexsort((weights, values), axis=diff_ax)
            values = np.take_along_axis(values, inds, axis)
            weights = np.take_along_axis(weights, inds, axis)
        norm_weights = np.squeeze(
            weights / np.atleast_1d(weights.sum(axis=axis))[:, None]
        )
        wquants = np.cumsum(norm_weights, axis=axis) - 0.5 * norm_weights
        res = find_wquant(values, wquants, 0.5, axis=axis, use_np=use_np)
    else:
        values = tt.as_tensor_variable(values)
        weights = tt.as_tensor_variable(weights)
        # FIXME: Not sure how to get theano to check the if every time,
        # always ordering for now
        # if not tt.all(tt.ge(tt.extra_ops.diff(values), 0)):
        # NOTE: lexsort is a custom op implemented here, but might be included
        # in aesara later. Results are consistent with numpy.
        # It is slow, but argsort (from aesara is slow as well)
        inds = lexsort((weights, values), axis=diff_ax)[0]
        # inds = tt.argsort(values, axis=diff_ax)
        values = take_along_axis(values, inds, axis)
        weights = take_along_axis(weights, inds, axis)
        # values = values[inds]
        # weights = weights[inds]
        # norm_weights = weights / weights.sum()
        norm_weights = tt.squeeze(
            weights / ut.tt_atleast_1d(weights.sum(axis=axis))[:, None]
        )
        wquants = (
            tt.extra_ops.cumsum(norm_weights, axis=axis) - 0.5 * norm_weights
        )
        # exoplanet interp function requires specific shapes
        # 0-d (scalar) does not matter much but let's be consistent
        # res = regular_grid_interp([wquants], values, np.array([[0.5]]).T)[0]
        res = find_wquant(values, wquants, 0.5, axis=axis, use_np=use_np)

    return res


def find_wquant(vals, wquants, q, axis=None, use_np=True):

    # Fancy way to find inds above and below quantile
    diff_ax = axis if axis is not None else 0
    if use_np:
        ind0 = np.argmax(
            np.diff(np.sign(wquants - q), axis=diff_ax), axis=axis
        )
        ind1 = ind0 + 1
        inds = np.stack([ind0, ind1]).T

        # Get weighted quantiles (x) and associated values
        xs = np.take_along_axis(wquants, inds, axis).T
        ys = np.take_along_axis(vals, inds, axis).T
    else:
        qdiff = wquants - q
        signs = qdiff / tt.abs_(qdiff)
        ind0 = tt.argmax(tt.extra_ops.diff(signs, axis=diff_ax), axis=axis)
        ind1 = ind0 + 1
        inds = tt.stack([ind0, ind1]).T

        xs = take_along_axis(wquants, inds, axis).T
        ys = take_along_axis(vals, inds, axis).T

    v = ys[0] + (0.5 - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])

    return v


def wmad(
    values: Union[np.ndarray, TensorVariable, TensorConstant],
    weights: Union[np.ndarray, TensorVariable, TensorConstant],
    med: Optional[float] = None,
    use_np: bool = True,
    axis: Optional[int] = None,
) -> Union[float, TensorVariable, TensorConstant]:
    """
    Calculate a weighted median absolute deviation with numpy or theano.

    :param values: Sample values to get the median.
    :type values: np.ndarray
    :param weights: Weights associated to each value.
    :type weights: np.ndarray
    :return: Weighted median absolute deviation of the sample values.
    :rtype: float
    """
    if med is None:
        med = wmed(values, weights, use_np=use_np, axis=axis)

    if use_np:
        diffs = np.abs(np.squeeze(values - np.atleast_1d(med)[:, None]))
    else:
        # diffs = tt.abs_(values - med)
        diffs = tt.abs_(tt.squeeze(values - ut.tt_atleast_1d(med)[:, None]))

    return wmed(diffs, weights, use_np=use_np, axis=axis)


def roll_mean(
    v: Union[np.ndarray, TensorVariable, TensorConstant],
    w: Union[np.ndarray, TensorVariable, TensorConstant],
    use_np: bool = True,
    axis: Optional[int] = None,
) -> Union[
    tuple[np.ndarray, np.ndarray],
    tuple[
        Union[TensorVariable, TensorConstant],
        Union[TensorVariable, TensorConstant],
    ],
]:

    if axis not in [None, 1]:
        raise ValueError("Only None and 1 are currently supported for axis")
    if use_np:
        out = np.average(v, weights=w, axis=axis)
        neff = np.sum(w, axis=axis) ** 2 / np.sum(w ** 2, axis=axis)
        vdiff = np.squeeze((v - np.atleast_1d(out)[:, None]))
        outvar = (
            np.sum(w * vdiff ** 2, axis=axis)
            / np.sum(w, axis=axis)
            * neff
            / (neff - 1)
        )
        err = np.sqrt(outvar / neff)
    else:
        out = tt.sum(v * w, axis=axis) / tt.sum(w, axis=axis)
        # err = tt.std(v) * tt.sqrt(tt.sum((w / tt.sum(w)) ** 2))
        neff = tt.sum(w, axis=axis) ** 2 / tt.sum(w ** 2, axis=axis)
        # outvar = tt.sum(w * (v - out) ** 2) / tt.sum(w) * neff / (neff - 1)
        vdiff = tt.squeeze((v - ut.tt_atleast_1d(out)[:, None]))
        outvar = (
            tt.sum(w * vdiff ** 2, axis=axis)
            / tt.sum(w, axis=axis)
            * neff
            / (neff - 1)
        )
        err = tt.sqrt(outvar / neff)

    return out, err


def roll_med(
    v: Union[np.ndarray, TensorVariable, TensorConstant],
    w: Union[np.ndarray, TensorVariable, TensorConstant],
    use_np: bool = True,
    axis: Optional[int] = None,
) -> Union[
    tuple[np.ndarray, np.ndarray],
    tuple[
        Union[TensorVariable, TensorConstant],
        Union[TensorVariable, TensorConstant],
    ],
]:

    # if axis == 1:
    #     breakpoint()
    if axis not in [None, 1]:
        raise ValueError("Only None and 1 are currently supported for axis")
    if use_np:
        # Numpy tricks to ensure 1d and then remove extra dims if any
        in_sum = np.squeeze(w / np.atleast_1d(np.sum(w, axis=axis))[:, None])
        std_factor = np.sqrt(np.sum(in_sum ** 2, axis=axis))
    else:
        # std_factor = tt.sqrt(tt.sum((w / tt.sum(w)) ** 2))
        in_sum = tt.squeeze(
            w / ut.tt_atleast_1d(tt.sum(w, axis=axis))[:, None]
        )
        std_factor = tt.sqrt(tt.sum(in_sum ** 2, axis=axis))

    out = wmed(v, weights=w, use_np=use_np, axis=axis)
    err = (
        1.4826
        * wmad(v, weights=w, med=out, use_np=use_np, axis=axis)
        * 1.253
        * std_factor
    )

    return out, err


def apply_moving(val, window, time, weight, apply_func, use_np=True):

    # TODO: Half window
    if use_np:
        # These arrays are broadcast with new axis 0 (vect is along ax 1)
        tmat = np.broadcast_to(time, (time.size,) * 2)
        vmat = np.broadcast_to(val, (val.size,) * 2)
        wmat = np.broadcast_to(weight, (weight.size,) * 2)
        hi = time + window
        lo = time - window
        mask = np.logical_and(tmat > lo[:, None], tmat < hi[:, None])
        out, err = apply_func(vmat * mask, wmat * mask, use_np=use_np, axis=1)
    else:
        masks = []
        tmat = np.broadcast_to(time, (time.size,) * 2)
        hi = time + window
        lo = time - window
        masks = tt.as_tensor_variable(
            np.logical_and(tmat > lo[:, None], tmat < hi[:, None])
        )

        out, err = apply_func(
            val * masks, weight * masks, use_np=use_np, axis=1
        )

        # def inloop(mask):
        #     oi, ei = apply_func(val[mask], weight[mask], use_np=use_np)
        #     return oi, ei
        # # ivals = np.arange(len(time))
        # (out, err), _ = scan(fn=inloop, sequences=[masks])

    return out, err
