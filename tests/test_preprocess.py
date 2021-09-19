import brav0.preprocess as pp
import brav0.utils as ut
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def id_sheet_url():
    gid = "1WMuvP2DZmCCjAggPzeXYea6wk_I5SAxvdM9kah8CFko"
    csv_query = "gviz/tq?tqx=out:csv&sheet=0"
    return f"https://docs.google.com/spreadsheets/d/{gid}/{csv_query}"


@pytest.fixture
def sheet_id_col():
    return "ODOMETER"


@pytest.fixture
def sheet_rv_col():
    return ["RV"]


@pytest.fixture
def id_list_all_cols(id_sheet_url, sheet_id_col):
    id_list = ut.get_bad_id_list(id_sheet_url, id_col=sheet_id_col)
    return id_list


@pytest.fixture
def odo_test_df():
    return pd.DataFrame({"FILENAME": ["2444444_", "2222222_"]})


@pytest.fixture
def id_list_one_col(id_sheet_url, sheet_id_col, sheet_rv_col):
    id_list = ut.get_bad_id_list(
        id_sheet_url, id_col=sheet_id_col, check_cols=sheet_rv_col
    )
    return id_list


@pytest.fixture
def expected_check_all():
    return ["2222222", "2222223", "2222224"]


@pytest.fixture
def expected_check_one():
    return ["2222222", "2222224"]


def test_id_kept_all(id_list_all_cols, expected_check_all):
    assert id_list_all_cols == expected_check_all


def test_id_kept_one(id_list_one_col, expected_check_one):
    assert id_list_one_col == expected_check_one


def test_filter_odo(odo_test_df, id_list_all_cols):
    """Test that odometers that should be filtered are filtered"""

    mask = pp.mask_bad_ids(
        odo_test_df, id_list_all_cols, "FILENAME", filter_method="odo"
    )
    filt_df = odo_test_df[mask]

    assert (filt_df == odo_test_df.iloc[:1]).all().bool()


def test_filter_odo_type(odo_test_df, id_list_all_cols):
    """Test odometer-filtered datasete is still a DataFrame"""
    mask = pp.mask_bad_ids(
        odo_test_df, id_list_all_cols, "FILENAME", filter_method="odo"
    )
    filt_df = odo_test_df[mask]

    assert isinstance(filt_df, pd.DataFrame)


def test_filter_bad_method_noargs(odo_test_df, id_list_all_cols):
    def bad_func():
        return "Allo"

    with pytest.raises(ValueError):
        pp.mask_bad_ids(
            odo_test_df, id_list_all_cols, "FILENAME", filter_method=bad_func
        )


def test_filter_bad_method_str(odo_test_df, id_list_all_cols):

    with pytest.raises(ValueError):
        pp.mask_bad_ids(
            odo_test_df, id_list_all_cols, "FILENAME", filter_method="allo"
        )


def test_filter_bad_method_rtype(odo_test_df, id_list_all_cols):
    def bad_func(a, b):
        return "Allo", a, b

    with pytest.raises(TypeError):
        pp.mask_bad_ids(
            odo_test_df, id_list_all_cols, "FILENAME", filter_method=bad_func
        )


@pytest.fixture
def outlier_index():
    return 5


@pytest.fixture
def sigclip_col():
    return "vrad"


@pytest.fixture
def df_with_outlier(outlier_index, sigclip_col):
    data_dict = {
        sigclip_col: np.ones(100) * 100,
        "othercolumn": np.ones(100) * 1.1,
    }
    # add an outlier
    data_dict[sigclip_col][outlier_index] = 400.0
    return pd.DataFrame(data_dict)


def test_clip_mask_clip(df_with_outlier, sigclip_col, outlier_index):
    series = df_with_outlier[sigclip_col]
    mask = pp.get_clip_mask(series, 3.0)
    clipped_ind = np.argwhere(~mask).item()
    assert clipped_ind == outlier_index


def test_clip_mask_noclip(df_with_outlier, sigclip_col):
    series = df_with_outlier[sigclip_col]
    mask = pp.get_clip_mask(series, 100)
    assert np.all(mask)
