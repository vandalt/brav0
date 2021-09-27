import aesara_theano_fallback.tensor as tt
import brav0.rolling as br
import numpy as np
import pytest


# =============================================================================
# Test weighted median on small examples
# =============================================================================
def test_med_equal(simple_values, tol, round_equal):
    med_np = np.median(simple_values)
    med_ut = br.wmed(simple_values, np.ones_like(simple_values))

    # Interpolation might introduce some error
    # assert np.round(med_ut, tol) == med_np
    assert round_equal(med_ut, med_np, tol)


def test_wmed_simple(simple_values, simple_weights):

    # TODO: This compares with past result (2.8) from ut function,
    # should use inpepdendent value
    assert br.wmed(simple_values, simple_weights) == 2.8


def test_mad_equal(simple_values, tol, round_equal):
    mad_np = np.median(np.abs(simple_values - np.median(simple_values)))
    mad_ut = br.wmad(simple_values, np.ones_like(simple_values))

    # Interpolation might introduce some error
    # assert np.round(mad_ut, tol) == mad_np
    assert round_equal(mad_ut, mad_np, tol)


def test_wmad_simple(simple_values, simple_weights):

    # TODO: This compares with past result (2.8) from ut function,
    # should use inpepdendent value
    assert br.wmad(simple_values, simple_weights) == 0.5942857142857141


# =============================================================================
# Test rolling function shapes
# =============================================================================
@pytest.fixture
def roll_x():
    return np.linspace(0, 100, num=1000)


@pytest.fixture
def roll_y(roll_x):
    # values don't really matter for the tests we'll do
    return 0.5 * roll_x + np.sin(np.pi * roll_x / 5)


@pytest.fixture
def roll_yerr(roll_x):
    return 0.1 * np.sin(np.pi * roll_x / 5) + 0.001


@pytest.fixture
def roll_weight(roll_yerr):
    return roll_yerr ** -2


@pytest.fixture(params=[True, False])
def use_np(request):
    return request.param


@pytest.fixture(params=[br.roll_mean, br.roll_med])
def roll_result(roll_y, roll_x, roll_weight, use_np, request):
    if not use_np:
        roll_y = tt.as_tensor_variable(roll_y)
        roll_weight = tt.as_tensor_variable(roll_weight)
    val, err = br.apply_moving(
        roll_y, 5.0 / 2, roll_x, roll_weight, request.param, use_np=use_np
    )
    return val, err, request.param.__name__, use_np


def test_rolling_shape(roll_result, roll_y):

    if roll_result[-1]:
        ymod = roll_result[0]
    else:
        ymod = roll_result[0].eval()

    assert ymod.shape == roll_y.shape


def test_rolling_no_change(data_dir, roll_result, tol, round_equal):

    val, err, name, npval = roll_result

    if not npval:
        val = val.eval()
        err = err.eval()

    old_val, old_err = np.loadtxt(data_dir / f"{name}.txt", unpack=True)

    assert round_equal(old_val, val, tol) and round_equal(old_err, err, tol)
