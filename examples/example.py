# %% [markdown]
# # Example Zero-point fit
# While _brav0_ can be run from the command line with a config file, it can be useful to access the API directly to experiment or run a specific model. This example notebook shows a GP model with this workflow.

# %%
from pathlib import Path

import yaml
import numpy as np
import xarray as xr
import arviz as az
import brav0.preprocess as pp
import brav0.utils as ut
import matplotlib.pyplot as plt
from brav0 import model, plot, rmplanets
from brav0.io import save_df, source_tables

# %% [markdown]
# ## Configuration
# We can specify some configuration parameters here before running the model.

# %%
time_col = "rjd"
vrad_col = "vrad"
svrad_col = "svrad"
file_col = "RVFILE"  # Label of the file column in df after sourcing files
row_col = "ROW"  # Label of index column that gives row number
used_cols = [time_col, vrad_col, svrad_col]
snr_col = "SNRCOL"
snrgoal_col = "SNRGOAL"
obj_col = "OBJECT"
id_filter_col = "FILENAME"

pp_list = ["nan", "sort", "sigma", "ID", "SNR", "error"]

# Glob pattern for input data files (we expect multiple stars)
in_pattern = "/path/to/data/*.rdb"
out_dir = Path("/path/to/outdir")

# Pre-processing options
# Bad odometer sheet for SPIRou
bad_id_url = "https://docs.google.com/spreadsheets/d/1WMuvP2DZmCCjAggPzeXYea6wk_I5SAxvdM9kah8CFko/gviz/tq?tqx=out:csv&sheet=0"
# Sigma clipping
nsig_clip = 3.0
snr_frac = 0.7
err_quant_cut = 0.95
bin_data = True
extra_wmean_pairs = dict()
extra_wmean_pairs["fwhm"] = "sig_fwhm"
az_kwargs = dict(
    var_names=[
        "~pred",
        "~pred_std",
        "~bkg",
        "~rv_model",
        "~dvdt",
        "~curv",
        "~_wn",
        "~logsigma",
        "~logrho",
    ],
    filter_vars="like",
)

# Targets to use in calibration
target_list = [
    "Star1",
    "Star2",
    "Star3",
]
# Map Dataset object name to exoplanet archive names if required
# NOTE: GL -> GJ is done automatically, can override here if necessary
archive_dict = {
    "Star1A": "Star1 A",
}

obj_extra_info = {
    "Star1": {
        # This overrides the default reference for this system
        "reference": "Alternative 2020",
    },
    "Star2": {
        "known_pl": {
            "c": {
                "pl_tranmid": 2400000.00,
                "pl_tranmiderr1": 0.1,
                "pl_tranmiderr2": -0.15,
            }
        }
    },
}

kernel = "Matern32Term"
with open("parameters.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
model_parameters = config["model_parameters"]

# Monte-Carlo settings for known planets removal
mc_prop = True
mc_ndraws = 5000


# %%
data = source_tables(in_pattern, file_col=file_col, row_col=row_col)
print("Data loaded")
save_df(data, out_dir / "raw.csv")

# %%
data_raw = data.copy()
data_clean = pp.cleanup(
    data_raw,
    plist=pp_list,
    bad_id_url=bad_id_url,
    id_filter_col=id_filter_col,
    clip_col=vrad_col,
    nsig_clip=nsig_clip,
    time_col=time_col,
    group_col=file_col,
    equant_col=svrad_col,
    equant_cut=err_quant_cut,
    snr_col=snr_col,
    snr_goal_col=snrgoal_col,
    snr_frac=snr_frac,
    used_cols=used_cols,
)
data_obj_ind = pp.index_with_obj(data_clean, 1, force_self_mask=True)
data_no_bin = pp.keep_obj_list(data_obj_ind, target_list, obj_col=obj_col)
data_bin = pp.bin_dataset(
    data_no_bin, vrad_col, svrad_col, extra_pairs=extra_wmean_pairs
)
data = pp.add_archive_name(data_bin, obj_col, extra_maps=archive_dict)


# %%
obj_list = ut.get_obj_vals(data, unique=True)

# %%
print("Pre-processing done.")
save_df(data, out_dir / "preprocessed.csv", force=True)

# %%
archive_list = rmplanets.get_archive_list(data)
known_pl = rmplanets.get_known_planets(archive_list)
archive_names = rmplanets.get_archive_map(data, row_col=row_col)
data_with_pl = data.copy()


# %%
# Here we take config references if specified
ref_cfg = rmplanets.get_archive_ref_cfg(obj_extra_info, archive_names)
# If no reference was specified, this will get the default one
known_pl = rmplanets.select_refs(known_pl, ref_cfg)

# %%
# With this function, we could override archive parameter values
archive_info = rmplanets.get_custom_archive(obj_extra_info, archive_names)
known_pl = rmplanets.add_custom_archive(known_pl, archive_info)

# %%
# Once the "known" parameter values are set, we can remove planets with
# monte-carlo error propagation (or without)
data = rmplanets.remove_all_planets(
    data_with_pl,
    known_pl,
    obj_col,
    time_col,
    mc_propagate=mc_prop,
    ndraws=mc_ndraws,
)
data_no_pl = data.copy()
save_df(data_no_pl, out_dir / "no_planets.csv", force=True)
print("Planet removal done")

# %%
gpzp_model = model.GPModel(data, kernel, model_parameters)

# %%
map_soln = gpzp_model.optimize()
map_soln_dict = map_soln
map_soln = xr.Dataset(map_soln)
map_soln.to_netcdf(out_dir / "map.nc")

# %%
trace = gpzp_model.sample(start=map_soln_dict, tune=500, draws=500)
post = trace.posterior

# %%
post.to_netcdf(out_dir / "posterior.nc")

# %%
summary = az.summary(trace, **az_kwargs)
print(summary)

# %%
post = trace.posterior
plot.plot_trace(post, **az_kwargs, savepath=out_dir / "trace.pdf")
plt.show()

# %%
plot.plot_corner(post, savepath=out_dir / "corner.pdf", **az_kwargs)
plt.clf()
# plt.show()

# %%
# TODO: move below to pre-defined functions
keys = [k for k in map_soln.keys() if "gamma" in k]
map_offs = map_soln[keys].to_pandas()
map_offs.index = map_offs.index.str.split("_").str[0]
map_offs.index.name = obj_col
data_no_offsets_fit = data.copy()
data_no_offsets_fit.vrad = (data_no_offsets_fit.vrad - map_offs).astype(float)
flatpost = post.stack(draws=("chain", "draw"))
pred_values = flatpost.pred.values
pred_std_values = flatpost.pred_std.values


# %%
plt.figure(figsize=(10, 6))
plot.plot_all(data_no_offsets_fit, alpha=0.5)
for i in np.random.randint(pred_values.shape[1], size=200):
    plt.plot(gpzp_model.tpred, pred_values[:, i], color="C0", alpha=0.3)
plot.plot_pred(
    gpzp_model, map_soln["pred"].values, map_soln["pred_std"].values
)
plt.show()

# %%
mask = np.isin(gpzp_model.tpred, data_no_offsets_fit.rjd)
resids_rv = data_no_offsets_fit.vrad.values - map_soln["pred"].values[mask]

# %%
plt.errorbar(
    data_no_offsets_fit.rjd,
    data_no_offsets_fit.vrad,
    data_no_offsets_fit.svrad,
    fmt="o",
    capsize=2,
)
plt.errorbar(
    data_no_offsets_fit.rjd,
    resids_rv,
    data_no_offsets_fit.svrad,
    fmt="o",
    capsize=2,
)
plt.show()

# %%
plot.plot_pred(
    gpzp_model, map_soln["pred"].values, map_soln["pred_std"].values
)
