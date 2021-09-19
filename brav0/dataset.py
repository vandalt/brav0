import os
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from astropy.table import Table
from pandas.core.frame import DataFrame

import brav0.preprocess as pp
import brav0.utils as ut


class Dataset:
    def __init__(
        self,
        pattern: Union[str, Path],
        in_ext=".rdb",
        vrad_col: str = "vrad",
        svrad_col: str = "svrad",
        file_col: str = "RVFILE",
        row_col: str = "ROW",
        obj_col: str = "OBJECT",
        obj_list: Optional[list[str]] = None,
        file_obj_ind: int = 1,
        file_obj_sep: str = "_",
        id_filter_col: str = "FILENAME",
        nsig_clip: Optional[int] = None,
        snr_col: Optional[str] = None,
        snr_goal_col: Optional[str] = None,
        snr_frac: Optional[float] = None,
        bad_id_list: Optional[list[str]] = None,
        bad_id_url: Optional[str] = None,
        err_quant_cut: Optional[float] = None,
        pre_binned: bool = False,
    ) -> None:

        self.file_col = file_col
        self.row_col = row_col
        self.obj_col = obj_col
        self.id_filter_col = id_filter_col
        self.nsig_clip = nsig_clip
        self.snr_col = snr_col
        self.snr_goal_col = snr_goal_col
        self.snr_frac = snr_frac
        self.vrad_col = vrad_col
        self.svrad_col = svrad_col

        if (
            err_quant_cut is not None and 0.0 <= err_quant_cut < 1.0
        ) or err_quant_cut is None:
            self.err_quant_cut = err_quant_cut
        else:
            raise ValueError("err_quant_cut must be between 0 and 1")

        # Assess if can do bad ID filtering
        if bad_id_list is None and bad_id_url is not None:
            self.bad_id_list = ut.get_bad_id_list(bad_id_url)
        elif bad_id_list is not None and bad_id_url is None:
            self.bad_id_list = bad_id_list
        elif bad_id_list is None and bad_id_url is None:
            self.bad_id_list = None
        else:
            msg = "Only one of bad_id_list and bad_id_url can be provided"
            raise ValueError(msg)

        # If all ok, set snr cut
        if (
            self.snr_col is not None
            and self.snr_goal_col is not None
            and self.snr_frac is not None
        ):
            if 0.0 <= snr_frac <= 1.0:
                raise ValueError("snr_frac must be between 0 and 1")
            self.do_snr_cut = True
        # if some but not all are set, raise error
        elif (
            self.snr_col is not None
            or self.snr_goal_col is not None
            or self.snr_frac is not None
        ):
            msg = "All or none of snr_col, snr_goal_col, snr_frac must be set"
            raise ValueError(msg)
        else:
            self.do_snr_cut = False

        pattern = Path(pattern)

        if pattern.is_file():
            # If single file, assume it's a dataset and read it directly
            df = pd.read_csv(pattern, index_col=[0, 1])
        else:

            if pattern.is_dir():
                # Glob all files in directory
                flist = ut.pathglob(pattern / f"*{in_ext}")
            else:
                # Glob pattern directly
                flist = ut.pathglob(pattern)

            # Go from list of astropy-readable files to a pandas df
            df = self._flist_to_df(flist)

        self._df = df
        self.binned = pre_binned

        # Use object names as an index
        self.add_obj_col(file_obj_ind, file_obj_sep=file_obj_sep)
        if obj_list is not None:
            self.keep_obj_list(None)

        # Set pp flag
        self.pp_done = []
        self.pp_nrows = 0

    @property
    def df(self) -> DataFrame:
        return self._df

    @df.setter
    def df(self, df):
        if not self.df.columns.isin(df.columns).all():
            msg = "The new df should contain all current columns"
            raise ValueError(msg)
        self._df = df

    def _flist_to_df(self, flist: list[str]) -> DataFrame:

        # Load bunch of files to put in directory
        data_dict = dict()
        for fpath in flist:
            fbase = os.path.basename(fpath)
            ftype = os.path.splitext(fbase)[-1][1:]
            data_dict[fbase] = Table.read(fpath, format=ftype).to_pandas()

        return pd.concat(data_dict, names=[self.file_col, self.row_col])

    def add_obj_col(
        self,
        file_obj_ind: int,
        file_obj_sep: str = "_",
    ) -> None:
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

        df = self.df.copy()

        # NOTE: this is a strong assumption on raw format/filenames
        #       can think of ways  to improve it in the future
        if self.obj_col in df.columns:
            msg = f"There is already an {self.obj_col} column, doing nothing."
            warnings.warn(msg, category=RuntimeWarning)
            return

        # Get filenames in pandas index object and split/index to get obj names
        file_lvl = df.index.get_level_values(self.file_col)
        objects = (
            file_lvl.str.split(file_obj_sep)
            .str[file_obj_ind]
            .set_names(self.obj_col)
        )

        # Add object column and reset index
        df[self.obj_col] = objects
        df = df.reset_index()

        if (df.groupby(self.obj_col).nunique()[self.file_col] > 1).any():
            msg = "Dataframe should not have more than one file per object"
            raise ValueError(msg)

        df = df.set_index([self.obj_col, self.row_col])
        # Move file column at the end
        df[self.file_col] = df.pop(self.file_col)

        self.df = df

    def preprocess(self):

        mask_df = pd.DataFrame()
        if self.bad_id_list is not None:
            mask_df["bad_id"] = pp.mask_bad_ids(
                self.df, self.bad_id_list, self.id_filter_col
            )
        if self.nsig_clip is not None and self.nsig_clip > 0.0:
            mask_df["sig_clip"] = pp.mask_sig_clip(
                self.df,
                self.vrad_col,
                nsig=self.nsig_clip,
                group_name=self.obj_col,
            )
        if self.do_snr_cut:
            mask_df["snr"] = pp.mask_snr_cut(
                self.df, self.snr_col, self.snr_goal_col, self.snr_f
            )
        if self.err_quant_cut is not None:
            mask_df["equant"] = pp.mask_quant(
                self.df,
                self.svrad_col,
                self.err_quant_cut,
                group_name=self.obj_col,
            )

        neg_mask_df = ~mask_df
        neg_mask_df
        mask_final = mask_df.all(axis=1)
        neg_mask_final = ~mask_final

        # Add pp that was just performed to stats
        self.pp_nrows += neg_mask_final.sum()
        self.pp_done.extend(mask_df.columns.to_list())

        self.df_raw = self.df.copy()
        self.df = self.df[mask_final]

    def bin_dataset(self, extra_pairs: Optional[dict[str, str]] = None):

        self.df_pre_binning = self.df.copy()

        df = self.df
        vrad_cols = df.columns[
            df.columns.str.startswith(self.vrad_col)
        ].tolist()
        svrad_cols = df.columns[
            df.columns.str.startswith(self.svrad_col)
        ].tolist()
        wmean_pairs = dict(zip(vrad_cols, svrad_cols))

        if extra_pairs is not None:
            wmean_pairs = {**wmean_pairs, **extra_pairs}

        self.df = ut.get_binned_data(self.df, wmean_pairs=wmean_pairs)

        self.binned = True

    def keep_obj_list(self, obj_list: list[str]):

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

        if self.obj_col in self.df.index.names:
            self.df = self.df.loc[obj_list]
        elif self.obj_col in self.df.columns:

            obj_mask = self.df[self.obj_col].isin(obj_list)

            self.df = self.df[obj_mask]
        else:
            msg = (
                f"{self.obj_col} is not an index."
                "Trying to filter with columns"
            )
            warnings.warn(msg)
