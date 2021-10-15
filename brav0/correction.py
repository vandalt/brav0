from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.table import Table
from pandas.core.frame import DataFrame
from scipy.interpolate import interp1d

import brav0.utils as ut
from brav0.io import load_zp


def correct_dataset(
    zp_path: Union[Path, str],
    rv_pattern: Union[Path, str],
    ext: str = ".rdb",
    **correct_file_kwargs,
):

    flist = ut.generate_flist(rv_pattern, ext=ext)

    for fpath in flist:
        correct_file(zp_path, fpath, **correct_file_kwargs)


def correct_file(
    zp_path: Union[Path, str],
    rvfile: Union[Path, str],
    zp_version: Optional[str] = None,
    force: bool = True,
    save_bin: bool = False,
    save_full: bool = True,
    vrad_label: str = "vrad",
    svrad_label: str = "svrad",
    extra_pairs: Optional[dict[str, str]] = None,
):

    in_file = Path(rvfile)
    zp_path = Path(zp_path)

    # We need a zp version to append to corrected RV files
    if zp_version is None:
        # If long path, assume brav0's default structure to get version name
        if len(zp_path.parents) > 2:
            zp_version = "_".join(zp_path.parts[-3:-1])
        elif str(zp_path).count("_") == 1:
            zp_version = str(zp_path).split("_")[1]
        else:
            raise ValueError(
                "Unexpected zero-point path. Need regular brav0 output path "
                "or zpcorr_{id}.sv format."
            )

    out_file = in_file.parent / (
        in_file.stem + f"_{zp_version}" + in_file.suffix
    )

    if save_bin:
        if out_file.stem.startswith("lbl"):
            # Special naming scheme for lbl
            bin_stem = out_file.stem.replace("lbl", "lbl2", 1)
        else:
            bin_stem = out_file.stem + "_binned"

        bin_file = out_file.parent / (bin_stem + out_file.suffix)

        # Bin check is true if need a bin calculation (force checked later)
        bin_needed = not bin_file.is_file()
    else:
        bin_needed = False

    # Out check is true if we needed a full calculation
    out_needed = not out_file.is_file() and save_full  #

    # If out and bin are not needed and force is off, we're done
    if not (out_needed or bin_needed or force):
        print("All files exist. Nothing to do here.")
        return

    tbl = Table.read(in_file, format=in_file.suffix[1:])

    zpc = load_zp(zp_path)

    tbl_corr = apply_correction(zpc, tbl)

    if save_full:
        tbl_corr.write(out_file, overwrite=True)

    if save_bin:
        vrad_colnames = [
            cn for cn in tbl.colnames if cn.startswith(vrad_label)
        ]
        svrad_colnames = [
            cn for cn in tbl.colnames if cn.startswith(svrad_label)
        ]
        wmean_pairs = dict(zip(vrad_colnames, svrad_colnames))
        if out_file.stem.startswith("lbl"):

            # LBL files have their own extra columns
            wmean_pairs["per_epoch_DDV"] = "per_epoch_DDVRMS"
            wmean_pairs["per_epoch_DDDV"] = "per_epoch_DDDVRMS"
            wmean_pairs["fwhm"] = "sig_fwhm"

        wmean_pairs["zpc"] = "szpc"

        if extra_pairs is not None:
            wmean_pairs = {**wmean_pairs, **extra_pairs}

        tbl2_corr = ut.bin_tbl(tbl_corr, wmean_pairs)

        tbl2_corr.write(bin_file, overwrite=True)


def apply_correction(zpc: DataFrame, tbl: Table) -> Table:

    rjd, vrad, svrad = tbl["rjd"], tbl["vrad"], tbl["svrad"]
    rjd_zp, vrad_zp, svrad_zp = zpc[["rjd", "vrad", "svrad"]].values.T

    # Linear interpolation of zp and zp error
    vrad_fint = interp1d(rjd_zp, vrad_zp, fill_value="extrapolate")
    svrad_fint = interp1d(rjd_zp, svrad_zp, fill_value="extrapolate")
    vrad_zp_resamp = vrad_fint(rjd)
    svrad_zp_resamp = svrad_fint(rjd)

    # We return the same table with corrections
    tbl_corr = tbl.copy()

    # Correction of quantites and add ZP
    tbl_corr["vrad_pre_zpc"] = vrad.copy()
    tbl_corr["svrad_pre_zpc"] = svrad.copy()
    tbl_corr["vrad"] = vrad - vrad_zp_resamp
    tbl_corr["svrad"] = np.sqrt(svrad ** 2 + svrad_zp_resamp ** 2)
    # TODO: Keep pre-correction RVs as well
    tbl_corr["zpc"] = vrad_zp_resamp
    tbl_corr["szpc"] = svrad_zp_resamp

    return tbl_corr
