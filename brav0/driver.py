import pickle
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3_ext as pmx
import xarray as xr
from box import Box

import brav0.preprocess as pp
import brav0.utils as ut
from brav0 import io, model, plot, rmplanets
from brav0.correction import correct_dataset


def source(config: Box):
    data = io.source_tables(
        config.in_pattern, file_col=config.file_col, row_col=config.row_col
    )

    if config.verbose:
        print("Data loaded")

    if config.out_dir.exists() and not config.force:
        raise FileExistsError(
            f"Output directory {config.out_dir} already exists."
            "Use -f/--force to overwrite anyway."
        )
    config.out_dir.mkdir(parents=True, exist_ok=config.force)
    io.save_df(data, config.out_dir / "raw.csv", force=config.force)

    if config.verbose:
        print("Raw Data saved")


def preprocess(config: Box):

    data_raw = io.load_df(config.out_dir / "raw.csv")

    if config.verbose:
        print("Raw dataframe loaded")

    # Extra nan filtering to plot raw data (helpful for raw series comparison)
    data_no_nan = pp.filter_nan_values(data_raw, used_cols=config.used_cols)
    plot.plot_all_objects(
        data_no_nan,
        ocol=config.file_col,
        tcol=config.time_col,
        rvcol=config.vrad_col,
        ervcol=config.svrad_col,
        out_dir=config.out_dir / "raw_plots",
        orientation="vertical",
        show=config.show,
    )

    data_clean = pp.cleanup(
        data_raw,
        plist=config.pp_list,
        bad_id_url=config.bad_id_url,
        id_filter_col=config.id_filter_col,
        clip_col=config.vrad_col,
        nsig_clip=config.clip_nsig,
        time_col=config.time_col,
        group_col=config.file_col,
        equant_col=config.svrad_col,
        equant_cut=config.err_quant_cut,
        snr_col=config.snr_col,
        snr_goal_col=config.snrgoal_col,
        snr_frac=config.snr_frac,
        used_cols=config.used_cols,
    )
    # TODO: Move this to config
    data_obj_ind = pp.index_with_obj(data_clean, 1, force_self_mask=True)
    data_no_bin = pp.keep_obj_list(
        data_obj_ind, config.target_list, obj_col=config.obj_col
    )
    if config.bin_data:
        data_bin = pp.bin_dataset(
            data_no_bin,
            config.vrad_col,
            config.svrad_col,
            extra_pairs=config.extra_wmean_pairs,
        )
    data = pp.add_archive_name(
        data_bin, config.obj_col, extra_maps=config.archive_dict
    )

    if config.verbose:
        print("Preprocessing done")

    summary = data.groupby(config.obj_col).apply(ut.get_summary)
    print("ZP Training data summary")
    print(summary)
    print()

    plot.plot_all_objects(
        data,
        ocol=config.obj_col,
        tcol=config.time_col,
        rvcol=config.vrad_col,
        ervcol=config.svrad_col,
        out_dir=config.out_dir / "pp_plots",
        orientation="vertical",
        show=config.show,
    )

    io.save_df(data, config.out_dir / "preprocessed.csv", force=config.force)
    if config.verbose:
        print("Preprocessed data saved")


def remove_planets(config: Box):

    data = io.load_df(config.out_dir / "preprocessed.csv")

    # Load archive info and do Monte-carlo
    archive_list = rmplanets.get_archive_list(data)
    known_pl = rmplanets.get_known_planets(archive_list)
    archive_names = rmplanets.get_archive_map(data, row_col=config.row_col)
    ref_cfg = rmplanets.get_archive_ref_cfg(
        config.obj_extra_info, archive_names
    )
    # If no reference was specified, this will get the default one
    known_pl = rmplanets.select_refs(known_pl, ref_cfg)
    archive_info = rmplanets.get_custom_archive(
        config.obj_extra_info, archive_names
    )
    known_pl = rmplanets.add_custom_archive(known_pl, archive_info)
    data = rmplanets.remove_all_planets(
        data,
        known_pl,
        config.obj_col,
        config.time_col,
        mc_propagate=config.mc_prop,
        ndraws=config.mc_ndraws,
    )
    print("The following known planets were removed:")
    print(list(known_pl["pl_name"]))

    io.save_df(data, config.out_dir / "no_planets.csv", force=config.force)

    summary = data.groupby(config.obj_col).apply(ut.get_summary)
    print("ZP Training data summary (planets removed)")
    print(summary)
    print()

    plot.plot_all_objects(
        data,
        ocol=config.obj_col,
        tcol=config.time_col,
        rvcol=config.vrad_col,
        ervcol=config.svrad_col,
        out_dir=config.out_dir / "no_planets",
        orientation="vertical",
        show=config.show,
    )


def model_zp(config: Box):

    data = io.load_df(config.out_dir / "no_planets.csv")

    # TODO: Write model info in directory, to make sure cli args taken into
    #       Or just don't allow cli args
    if config.model_subdir is None:
        model_subdir = config.model
    else:
        model_subdir = config.model_subdir

    model_dir = ut.make_unique_dir(config.out_dir / model_subdir)

    model_parameters = ut.get_config_params(config)

    if config.model == "Matern32":
        zpmodel = model.GPModel(
            data,
            "Matern32Term",
            model_parameters,
            time_label=config.time_col,
            vrad_label=config.vrad_col,
            svrad_label=config.svrad_col,
            obj_label=config.obj_col,
        )
    elif config.model == "SumMatern32":
        zpmodel = model.GPModel(
            data,
            "SumMatern32Term",
            model_parameters,
            time_label=config.time_col,
            vrad_label=config.vrad_col,
            svrad_label=config.svrad_col,
            obj_label=config.obj_col,
        )
    elif config.model == "ExpQuad":
        zpmodel = model.GPModel(
            data,
            "ExpQuad",
            model_parameters,
            time_label=config.time_col,
            vrad_label=config.vrad_col,
            svrad_label=config.svrad_col,
            obj_label=config.obj_col,
        )
    elif config.model.lower() == "rolling":
        zpmodel = model.RollingModelPymc(
            data,
            config.roll_window,
            model_parameters,
            method=config.roll_method,
            time_label=config.time_col,
            vrad_label=config.vrad_col,
            svrad_label=config.svrad_col,
            obj_label=config.obj_col,
        )
    else:
        msg = (
            f"Model {config.model} is not supported."
            " This should have been caught by the CLI interface."
            " Please consider opening an issue on Github:"
            " https://github.com/vandalt/brav0/issues."
        )
        raise ValueError(msg)

    tpred = zpmodel.tpred
    if config.optimize:
        map_soln = zpmodel.optimize()
        ut.save_map_dict(map_soln, model_dir / "map.pickle")
        map_pred = map_soln["pred"]
        map_pred_std = map_soln["pred_std"]
        map_pred_df = pd.DataFrame(
            {
                config.time_col: tpred,
                config.vrad_col: map_pred,
                config.svrad_col: map_pred_std,
            }
        )
        map_pred_df.to_csv(model_dir / "zp_map.csv", index=False)
    if config.sample:
        trace = zpmodel.sample(
            start=map_soln if config.optimize else None,
            tune=config.tune,
            draws=config.draws,
        )
        post = trace.posterior
        post.to_netcdf(model_dir / "posterior.nc")
        # Get flattened posterior
        flatpost = post.stack(draws=("chain", "draw"))
        # Get quantiles
        flatpost_quant = flatpost.quantile([0.16, 0.50, 0.84], dim="draws")
        pred_16, pred_med, pred_84 = flatpost_quant["pred"].values
        # Get uncertinaty from mcmc draws
        pred_lo = pred_med - pred_16
        pred_hi = pred_84 - pred_med
        pred_std_mcmc = np.mean([pred_lo, pred_hi], axis=0)
        # Get median model envelope
        pred_std_med = flatpost_quant["pred_std"].values[1]
        # Save with MCMC uncertaint
        pred_med_df_mcmc = pd.DataFrame(
            {
                config.time_col: tpred,
                config.vrad_col: pred_med,
                config.svrad_col: pred_std_mcmc,
            }
        )
        pred_med_df_mcmc.to_csv(model_dir / "zp_med_mcmc.csv", index=False)
        pred_med_df_model = pd.DataFrame(
            {
                config.time_col: tpred,
                config.vrad_col: pred_med,
                config.svrad_col: pred_std_med,
            }
        )
        pred_med_df_model.to_csv(model_dir / "zp_med_model.csv", index=False)

    if not (config.optimize or config.sample):

        # If not optimization or sampling, use test values to evaluate model
        # and offsets

        off_keys = ut.get_substr_keys("gamma", zpmodel)
        off_vars = [zpmodel[k] for k in off_keys]
        off_values = np.array(pmx.eval_in_model(off_vars, model=zpmodel))
        off_series = pd.Series(off_values, index=off_keys)
        off_series.index = off_series.index.str.split("_").str[0]
        off_series.index.name = "OBJECT"
        off_series.to_csv(model_dir / "offsets_test.csv")
        pred = pmx.eval_in_model(zpmodel.pred, model=zpmodel)
        pred_std = pmx.eval_in_model(zpmodel.pred_std, model=zpmodel)
        pred_test = pd.DataFrame(
            {
                config.time_col: tpred,
                config.vrad_col: pred,
                config.svrad_col: pred_std,
            }
        )
        pred_test.to_csv(model_dir / "zp_test.csv", index=False)


def summary(config: Box):

    modeldir = Path(config.modeldir)
    map_path = modeldir / "map.pickle"
    post_path = modeldir / "posterior.nc"
    test_path = modeldir / "zp_test.csv"

    data = io.load_df(
        config.out_dir / "no_planets.csv", sort_col=config.time_col
    )

    if post_path.is_file():
        post = xr.open_dataset(modeldir / "posterior.nc")

        az_summary = az.summary(post, **config.arviz_kwargs)
        print(az_summary)

        # MCMC plots
        # Trace
        plot.plot_trace(
            post, **config.arviz_kwargs, savepath=modeldir / "trace.pdf"
        )
        if config.show:
            plt.show()
        else:
            plt.close()

        # Corner plot
        plot.plot_corner(
            post,
            savepath=modeldir / "corner.pdf",
            **config.arviz_kwargs,
            plot_datapoints=False,  # Many parameters: datapoints crash corner
        )
        if config.show:
            plt.show()
        else:
            plt.close()

        # Load zero-point
        zpcurve = pd.read_csv(modeldir / "zp_med_model.csv")

        # Subtract offsets from data
        flatpost = post.stack(draws=("chain", "draw"))
        # Get quantiles
        flatpost_med = flatpost.median(dim="draws")

        # Get offsets and sbutract
        off_keys = ut.get_substr_keys("gamma", post=post)
        offsets = flatpost_med[off_keys].to_pandas()
        offsets.index = offsets.index.str.split("_").str[0]
        offsets.index.name = "OBJECT"

        # Get wn for record
        wn_keys = ut.get_substr_keys("_wn", post=post)
        wns = flatpost_med[wn_keys].to_pandas()
        wns.index = wns.index.str.split("_").str[0]
        wns.index.name = "OBJECT"

    elif map_path.is_file():
        # Load MAP and posterior
        with open(modeldir / "map.pickle", "rb") as pfile:
            map_soln = pickle.load(pfile)

        off_keys = ut.get_substr_keys("gamma", map_dict=map_soln)
        offsets = pd.Series({k: map_soln[k] for k in off_keys})
        offsets.index = offsets.index.str.split("_").str[0]
        offsets.index.name = "OBJECT"

        # Get wn for record
        wn_keys = ut.get_substr_keys("_wn", map_dict=map_soln)
        wns = pd.Series({k: map_soln[k] for k in wn_keys})
        wns.index = wns.index.str.split("_").str[0]
        wns.index.name = "OBJECT"

        zpcurve = pd.read_csv(modeldir / "zp_map.csv")

    elif test_path.is_file():
        offsets = pd.read_csv(
            modeldir / "offsets_test.csv", squeeze=True, index_col=0
        )
        zpcurve = pd.read_csv(modeldir / "zp_test.csv")
    else:
        raise FileNotFoundError(
            "There does not seem to be any model output to summarize."
        )

    data_no_offsets = data.copy()
    data_no_offsets.vrad = (data_no_offsets.vrad - offsets).astype(float)
    # TODO: Get residuals (from function, and save)
    mask = np.isin(zpcurve[config.time_col], data_no_offsets[config.time_col])
    resids_rv = (
        data_no_offsets[config.vrad_col].values
        - zpcurve.loc[mask, config.vrad_col]
    ).values
    resids_df = data_no_offsets.copy()
    resids_df[config.vrad_col] = resids_rv

    # Plot data, model and residuals
    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0},
    )

    axrv, axres = axs
    plot.plot_all(data_no_offsets, ax=axrv, alpha=0.6)
    if (
        not zpcurve[config.time_col]
        .isin(data_no_offsets[config.time_col])
        .all()
    ):
        plot.plot_pred(
            zpcurve[config.time_col],
            zpcurve[config.vrad_col],
            zpcurve[config.svrad_col],
            ax=axrv,
        )
    else:
        axrv.errorbar(
            zpcurve[config.time_col],
            zpcurve[config.vrad_col],
            yerr=zpcurve[config.svrad_col],
            fmt="r^",
            markersize=3,
            capsize=2,
            zorder=100,
        )
    axrv.set_ylabel("RV [m/s]")
    # Plot residuals
    axres.axhline(0.0, linestyle="--", color="r")
    plot.plot_all(resids_df, ax=axres)
    nobj = len(data.groupby("OBJECT"))
    axrv.legend(ncol=nobj // 5)
    axres.set_ylabel("O-C [m/s]")
    axres.set_xlabel("RJD")
    plt.tight_layout()
    plt.savefig(modeldir / "pred.pdf")
    if config.show:
        plt.show()
    else:
        plt.close(fig)

    plot.plot_all_objects(
        data_no_offsets,
        ocol=config.obj_col,
        tcol=config.time_col,
        rvcol=config.vrad_col,
        ervcol=config.svrad_col,
        out_dir=modeldir / "object_plots_data",
        orientation="vertical",
        show=config.show,
    )
    plot.plot_all_objects(
        resids_df,
        ocol=config.obj_col,
        tcol=config.time_col,
        rvcol=config.vrad_col,
        ervcol=config.svrad_col,
        out_dir=modeldir / "object_plots_residuals",
        orientation="vertical",
        show=config.show,
    )

    try:
        ut.print_data_info(data, config, wn_dict=wns)
    except NameError:
        ut.print_data_info(data, config)

    # ZP periodogram
    zpcurve_in_data = zpcurve[mask]
    for zp, lab in zip([zpcurve, zpcurve_in_data], ["full", "in_data"]):
        freq, pwr = plot.plot_series_periodogram(
            zp[config.time_col],
            zp[config.vrad_col],
            zp[config.svrad_col],
            orientation="vertical",
        )
        plt.tight_layout()
        plt.savefig(modeldir / f"perio_{lab}.pdf")
        np.savetxt(modeldir / f"perio_{lab}.txt", np.array([freq, pwr]).T)
        if config.show:
            plt.show()
        else:
            plt.close(fig)

        print(f"Info for {lab} correction")
        print(f"  Mean model error: {np.mean(zp[config.svrad_col])}")
        print(f"  Median model error: {np.median(zp[config.svrad_col])}")


def correct(config: Box):
    if config.verbose:
        print(f"Starting correction of files matching {config.rvpattern}")
    if not config.ext.startswith("."):
        config.ext = "." + config.ext
    if config.rv_pattern.startswith("~"):
        config.rv_pattern = str(Path.home() / config.rv_pattern[2:])
    correct_dataset(
        config.zpc_path,
        config.rv_pattern,
        ext=config.ext,
        zp_version=config.zp_version,
        force=config.force,
        save_bin=config.save_bin,
        save_full=config.save_full,
        vrad_label=config.vrad_col,
        svrad_label=config.svrad_col,
        extra_pairs=config.extra_wmean_pairs,
    )
