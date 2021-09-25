import re
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
from astroquery.utils.tap.core import TapPlus
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from radvel.kepler import rv_drive
from radvel.orbit import timetrans_to_timeperi
from scipy.stats import norm, truncnorm

ARCHIVE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP"

# Keys from the archive
PER_KEY = "pl_orbper"
TP_KEY = "pl_orbtper"
TC_KEY = "pl_tranmid"
ECC_KEY = "pl_orbeccen"
OMEGA_KEY = "pl_orblper"
K_KEY = "pl_rvamp"
TRANSIT_FLAG = "tran_flag"

# Keys used to model orbit
ORB_KEYS = [
    PER_KEY,
    TP_KEY,
    ECC_KEY,
    OMEGA_KEY,
    K_KEY,
]


def get_archive_list(data: DataFrame) -> list[str]:
    """
    Get unique list of archive names in a dataset

    :param data: Dataset
    :type data: DataFrame
    :return: List of archive names
    :rtype: list[str]
    """
    return list(data.ARCHIVE.unique())


def get_archive_map(data: DataFrame, row_col: Optional[str] = "ROW") -> Series:
    """
    Get a series mapping object names to archive names

    :param data: Dataset with archive names as ARCHIVE column and object names
                 in index
    :type data: DataFrame
    :param row_col: column with rol index, defaults to "ROW". Set to None if
                    not applicable
    :type row_col: str, optional
    :return: Series mapping object names to archive names
    :rtype: Series
    """
    archive_map = data.ARCHIVE.drop_duplicates()
    if row_col is not None:
        archive_map = archive_map.droplevel(row_col)
    return archive_map


def get_archive_ref_cfg(
    info: dict[str, dict], archive_map: Mapping[str, str]
) -> Series:
    """
    Get mapping from archive name to refrence, using a config dict.

    :param info: Info configuration dictionary with object names as keys
    :type info: dict[str, dict]
    :param archive_map: Mapping from object names to archive names
    :type archive_map: Mapping[str, str]
    :return: [TODO:description]
    :rtype: Series
    """
    ref_cfg = dict(
        [
            (objname, info["reference"])
            for objname, info in info.items()
            if "reference" in info
        ]
    )
    ref_cfg = pd.Series(ref_cfg)
    ref_cfg.index = ref_cfg.index.map(archive_map)

    return ref_cfg


def get_known_planets(
    target_list: List, keep_controv: bool = False, default_only: bool = False
) -> DataFrame:
    """
    Get a DataFrame with known planets for a list of stars.
    This uses the exoplanet archive table:
    https://exoplanetarchive.ipac.caltech.edu

    :param target_list: List of valid archive names.
    :type target_list: List
    :param keep_controv: Keep planets flagged as controversial in the archive,
                         defaults to False
    :type keep_controv: bool, optional
    :param default_only: Keep only default parameter set for each planet,
                         defaults to False
    :type default_only: bool, optional
    :return: DataFrame with known planets loaded from the archive
    :rtype: DataFrame
    """

    # Some parameters to use in our SQL query
    table = "ps"
    host_key = "hostname"
    target_strings = [f"'{t}'" for t in target_list]
    targets = ", ".join(target_strings)

    archive = TapPlus(ARCHIVE_URL)
    sql_query = (
        f"SELECT * " f"FROM {table} " f"WHERE {host_key} IN ({targets})"
    )
    if not keep_controv:
        controv_flag = "pl_controv_flag"
        controv_val = int(keep_controv)
        sql_query = " ".join(
            [sql_query, f"AND {controv_flag} = {controv_val}"]
        )
    if default_only:
        default_flag = "default_flag"
        default_val = int(default_only)
        sql_query = " ".join(
            [sql_query, f"AND {default_flag} = {default_val}"]
        )

    # We use async job to allow > 2000 lines in astroquery, just in case
    job = archive.launch_job_async(sql_query)
    df = job.get_results().to_pandas()

    return df


def select_refs(df: DataFrame, ref_cfg: Mapping):
    """
    Select reference from archive table for each object.

    :param df: DataFrame with archive table
    :type df: DataFrame
    :param ref_cfg: Mapping of object to reference
    :type ref_cfg: Mapping
    """

    ref_obj = list(ref_cfg.keys())
    objects = df.hostname.unique()

    out_list = []
    for obj in objects:
        df_obj = df[df.hostname == obj]
        if obj in ref_obj:
            regex = "[^a-zA-Z0-9.*]"
            ref_pattern = re.sub(regex, ".*", ref_cfg[obj].replace(" ", ".*"))
            out_list.append(
                df_obj[df_obj.pl_refname.str.contains(ref_pattern)]
            )
        else:
            out_list.append(df_obj[df_obj.default_flag.astype(bool)])

    return pd.concat(out_list, ignore_index=True)


def get_custom_archive(
    info: dict[str, dict], archive_map: Mapping[str, str]
) -> dict[str, dict]:
    """
    Convert custom archive dict from using object name to using archive name.

    :param info: Info configuration dictionary with object names as keys
    :type info: dict[str, dict]
    :param archive_map: Mapping from object names to archive names
    :type archive_map: Mapping[str, str]
    :return: Dictionary mapping archive name to custom archive parameter
    :rtype: Series
    """
    # TODO: This is dup code from get_archive_ref_cfg, merge it
    archive_info = dict(
        [
            (archive_map[oname], info[oname])
            for oname in archive_map.index
            if oname in info
        ]
    )
    return archive_info


def add_custom_archive(df: DataFrame, target_info: Mapping):
    """
    Allow users to supplement/override the archive table with custom
    informations.

    :param df: DataFrame with the archive table
    :type df: DataFrame
    :param target_info: Mapping with extra parameters for each object.
    :type target_info: Mapping
    """

    df = df.copy()

    # TODO: Add safety here to make sure required values are there
    # And that provided values have no error in name
    for obj in target_info:
        try:
            extra_pl = target_info[obj]["known_pl"]
        except KeyError:
            continue
        for pl_letter in extra_pl:
            # NOTE: Cannot just check pl_name from archive, not always same
            # hostname
            pl_mask = (df.hostname == obj) & (df.pl_letter == pl_letter)
            if pl_mask.any():
                row = df[pl_mask]
                pl_params = extra_pl[pl_letter]
                for label, val in pl_params.items():
                    row[label] = val
                df[pl_mask] = row
            else:
                row = pd.Series(extra_pl[pl_letter])
                row["pl_name"] = " ".join([obj, pl_letter])
                row["pl_letter"] = pl_letter
                row["hostname"] = obj
                df.append(row)

    return df


def draw_param(planet: Series, key: str, ndraws: int) -> np.ndarray:
    """
    Draw parameter values based on planet parameters and uncertainties.

    :param planet: Pandas series with planet parameters.
    :type planet: Series
    :param key: Key of the parameter to draw
    :type key: str
    :param ndraws: Number of draws.
    :type ndraws: int
    :return: Array of Monte-Carlo draws for the parameter.
    :rtype: np.ndarray
    """

    pval = planet[key]
    err = np.mean(np.abs([planet[key + f"err{i}"] for i in (1, 2)]))

    if np.any(np.isnan([pval, err])):
        raise ValueError(f"Value and error for {key} must not be NaN")

    if key in [ECC_KEY, PER_KEY, K_KEY]:
        # Truncated normal if unphysical below 0
        upper = (1.0 - pval) / err if key == ECC_KEY else np.inf
        a, b = (0 - pval) / err, upper
        dist = truncnorm(a, b, loc=pval, scale=err)
    else:
        dist = norm(loc=pval, scale=err)

    return dist.rvs(ndraws)


def get_tp_pl(planet: Series, ndraws: int = 0) -> Union[float, np.ndarray]:
    """
    Get the time of periastron for a given planet

    :param planet: Pandas series with planet parameters
    :type planet: Series
    :param ndraws: Number of Monte-Carlo draws to use,
                   defaults to 0 (deterministic).
    :type ndraws: int, optional
    :return: Time of periastron for the planet, either as a single value or
             an array of draws.
    :rtype: float
    """

    def from_tc(planet, ndraws=0):
        if ndraws > 0:
            tc = draw_param(planet, TC_KEY, ndraws)
        else:
            tc = planet[TC_KEY]
        tp = timetrans_to_timeperi(
            tc, planet[PER_KEY], planet[ECC_KEY], planet[OMEGA_KEY]
        )
        return tp

    def from_tp(planet, ndraws=0):
        if ndraws > 0:
            tp = draw_param(planet, TP_KEY, ndraws)
        else:
            tp = planet[TP_KEY]
        return tp

    if planet[TRANSIT_FLAG]:
        # For transiting planets, Tc is usually better constained -> try first
        try:
            tp = from_tc(planet, ndraws=ndraws)
        except (IndexError, ValueError):
            tp = from_tp(planet, ndraws=ndraws)
    else:
        try:
            tp = from_tp(planet, ndraws=ndraws)
        except (IndexError, ValueError):
            tp = from_tc(planet, ndraws=ndraws)
    return tp


def get_orbit_params(
    planet: Series, prop_dict: Dict[str, bool], ndraws: int = 5000
) -> Series:
    """
    Get orbital parameters for a planet, either as values or as Monte-Carlo
    samples based on archive values and uncertainties.

    :param planet: Pandas series with parameters for a planet from the archive
    :type planet: Series
    :param prop_dict: Mapping indicating if errors should be propagated with
                      MC for each parameters.
    :type prop_dict: Dict[str, bool]
    :param ndraws: Number of MC draws, defaults to 5000
    :type ndraws: int, optional
    :return: Returns series of orbital parameters. One float per parameter if
             no MC draw, or an array otherwise.
    :rtype: Series
    :raises ValueError: Raises a ValueError if a parameter is missing (if Tp
                        is missing, Tc is required).
    """

    # Convert units at the beginning
    planet_out = planet.copy()
    # omega and its errors
    omega_keys = [OMEGA_KEY] + [OMEGA_KEY + f"err{i}" for i in (1, 2)]
    # Because planet_out has strings convert to float before np.radians
    planet_out[omega_keys] = np.radians(planet_out[omega_keys].astype(float))

    for key in ORB_KEYS:
        if key == TP_KEY:
            if planet_out[[TP_KEY, TC_KEY]].isna().all():
                raise ValueError(
                    f"Both {TP_KEY} and {TC_KEY} are NaN"
                    f" for {planet_out.pl_name}"
                )
            # Tp is a special case because we might need Tc for best precision
            # When we draw MC parameters, we need to draw all others before
            continue
        else:
            if np.isnan(planet_out[key]):
                raise ValueError(
                    f"Parameter {key} for {planet_out.pl_name} is NaN"
                )
            # We either draw or keep the same value
            if prop_dict[key]:
                planet_out[key] = draw_param(planet_out, key, ndraws)

    ndraws_tp = ndraws if prop_dict[TP_KEY] else 0
    planet_out[TP_KEY] = get_tp_pl(planet_out, ndraws=ndraws_tp)

    # Make sure all arrays or only integers
    params = planet_out[ORB_KEYS]
    try:
        lvals = params.str.len()
        plen = int(lvals.max())  # All non-zero should be max
        scalar_mask = lvals.isna()
        params[scalar_mask] = params[scalar_mask].apply(
            lambda x: np.full(plen, x)
        )
    except AttributeError:
        # if all scalars, nothing to do (just filter to dict)
        pass

    return params


def remove_one_planet(
    data: DataFrame,
    planet: Series,
    t_label: str,
    mc_propagate: Union[bool, List[bool]] = True,
    ndraws: int = 5000,
) -> DataFrame:
    """
    Remove a planet from an RV dataset using the exoplanet archive.

    :param data: Dataframe with RV data for a single object (date-compatible)
    :type data: DataFrame
    :param planet: Row from the exoplanet archive with planet information,
                   stored as a pandas series.
    :type planet: Series
    :param t_label: Label of the time column in our data
    :type t_label: str
    :param mc_propagate: Propgate Monte-Carlo error from table.
                         boolean or list of bool.
                         If a list, order is [per, tp, ecc, omega, k].
    :type mc_propagate: bool
    :return: Data with known planets removed
    :rtype: DataFrame
    """
    # Check if we do MC error propagation (yes, if any of mc_propagate is true)
    do_prop = np.any(mc_propagate)

    # Make a dict mapping orbit params keys MC propagation flag
    try:
        prop_dict = dict(zip(ORB_KEYS, mc_propagate))
    except TypeError:
        prop_dict = dict(zip(ORB_KEYS, [mc_propagate] * 5))
    params = get_orbit_params(planet, prop_dict=prop_dict, ndraws=ndraws)

    # make sure time is in the right frame for radvel rv_drive
    t = data[t_label].values  # rv_drive takes an array
    if t_label.upper() == "RJD":
        t = t + 2400000
    elif t_label not in ["JD", "BJD"]:
        raise ValueError(f"Unkown time type: {t_label}")

    if not do_prop:
        # NOTE: Order matters for radvel
        plist = [
            params[PER_KEY],
            params[TP_KEY],
            params[ECC_KEY],
            params[OMEGA_KEY],
            params[K_KEY],
        ]
        kep_orb = rv_drive(t, plist)
        data.vrad = data.vrad - kep_orb
    else:
        # Stacking the series gives one row per paramter, columns are draws
        params = params[ORB_KEYS]  # Order for radvel
        parr = np.stack(params)
        # Iterate over draws (columns)
        kep_draws = np.zeros([ndraws, t.size])
        for i, orbpar in enumerate(parr.T):
            kep_draws[i] = rv_drive(t, orbpar)

        # Get curve and enveloppe from draws
        kep_16th, kep_med, kep_84th = np.percentile(
            kep_draws, [16, 50, 84], axis=0
        )
        kep_err_lo = kep_med - kep_16th
        kep_err_hi = kep_84th - kep_med
        kep_err = np.mean([kep_err_hi, kep_err_lo])

        # Substract and propagate the MC enveloppe
        data.vrad = data.vrad - kep_med
        data.svrad = np.sqrt(data.svrad ** 2 + kep_err ** 2)

    return data


def remove_all_planets(
    data: DataFrame,
    known_df: DataFrame,
    obj_label: str,
    t_label: str,
    ndraws: int = 5000,
    mc_propagate: Union[bool, List[bool]] = True,
) -> DataFrame:
    """
    Remove all known planets for all objects.

    :param data: Data with planets for all objects.
    :type data: DataFrame
    :param known_df: DataFrame of known planets
    :type known_df: DataFrame
    :param obj_label: Label that gives the object name
    :type obj_label: str
    :param t_label: Label of the time column in our data
    :type t_label: str
    :param mc_propagate: Propgate Monte-Carlo error from table.
                         boolean or list of bool.
                         If a list, order is [per, tp, ecc, omega, k].
    :type mc_propagate: bool
    :param ndraws: Number of monte-carlo draws if mc_propagate is true
    :type ndraws: int
    :return: Data with known planets removed
    :rtype: DataFrame
    """
    df_list = []

    data = data.copy()

    for _, obj_data in data.groupby(obj_label):
        # Get known planets for this object
        known_df_obj = known_df[known_df.hostname == obj_data.ARCHIVE[0]]
        # Run removal for this object (re-use same name to remove ALL planets)
        for _, planet in known_df_obj.iterrows():
            obj_data = remove_one_planet(
                obj_data,
                planet,
                t_label,
                mc_propagate=mc_propagate,
                ndraws=ndraws,
            )
        df_list.append(obj_data)

    data_no_planets = pd.concat(df_list)

    return data_no_planets
