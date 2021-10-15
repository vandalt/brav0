from argparse import ArgumentError, ArgumentParser, Namespace
from pathlib import Path

from box import Box

import brav0.utils as ut
from brav0 import __version__, driver


def config_override(config: Box, clargs: Namespace) -> Box:
    argdict = vars(clargs)
    for arg in argdict:
        if (argdict[arg] is not None) or (
            argdict[arg] is None and arg not in config
        ):
            config[arg] = argdict[arg]
    return config


def main():
    # The main parser
    psr = ArgumentParser(
        description="brav0: Bayesian Radial velocity zero-point correction"
    )
    psr.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subpsr = psr.add_subparsers()

    # The parent parser to all subparsers with common options
    psr_parent = ArgumentParser(add_help=False)
    config_arg = psr_parent.add_argument(
        "config", type=str, help="Configuration file."
    )
    psr_parent.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,  # Override False as default even if store true
        help="Verbose mode (off by default)",
    )
    psr_parent.add_argument(
        "-o",
        "--output-directory",
        dest="out_dir",
        type=str,
        default=None,
        help="Output directory where everything related to a ZP run is saved",
    )
    psr_parent.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite if the directory or file already exists.",
    )
    psr_parent.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Show summary plots in addition to saving them.",
    )

    def _new_subpsr(name, **kwargs):
        return subpsr.add_parser(name, parents=[psr_parent], **kwargs)

    # Subparser to load a dataset and save it
    psr_source = _new_subpsr(
        "source",
        description="Source RV data and save corresponding Zero-point dataframe",
    )
    psr_source.set_defaults(func=driver.source)

    # Subparser to preprocess data
    psr_pp = _new_subpsr(
        "preprocess",
        description="Preprocess data and the cleaned data to csv.",
    )
    psr_pp.set_defaults(func=driver.preprocess)

    # Subparser to remove known planets
    psr_planets = _new_subpsr(
        "remove-planets",
        description="Remove known planets using data from the exoplanet archive.",
    )
    psr_planets.set_defaults(func=driver.remove_planets)

    # Subparser for ZP modelling
    psr_model = _new_subpsr(
        "model",
        description="Model zero-point",
    )
    psr_model.add_argument(
        "model",
        type=str,
        choices=["Matern32", "SumMatern32", "ExpQuad", "Rolling"],
        help="Zero-point model name.",
    )
    psr_model.add_argument(
        "-m" "--roll-method",
        dest="roll_method",
        type=str,
        default="wmean",
        choices=["mean", "median", "wmean", "wmedian"],
        help="Method to calculate rolling values.",
    )
    subdir_help = (
        "Subdirectory where the model output is stored."
        " By default, this will be model name. If a directory exists,"
        " the full time (_YYMMDDHHMMSS) will be appended to ensure uniqueness."
    )
    psr_model.add_argument(
        "-d" "--model-subdir",
        dest="model_subdir",
        type=str,
        default=None,
        help=subdir_help,
    )
    psr_model.add_argument(
        "-w" "--roll-window",
        dest="roll_window",
        type=float,
        default=7.0,
        help="Width of the rolling window in days.",
    )
    psr_model.set_defaults(func=driver.model_zp)

    psr_summary = _new_subpsr(
        "summary",
        description="Generate summary plots and information for a given ZP model",
    )
    psr_summary.add_argument(
        "modeldir", help="Directory where the model results are saved."
    )
    psr_summary.set_defaults(func=driver.summary)

    psr_correct = _new_subpsr(
        "correct", description="Apply the ZPC to one or multiple RV datasets."
    )
    psr_correct.add_argument(
        "zpc_path",
        help="Path to the ZPC csv file",
    )
    psr_correct.add_argument(
        "rv_pattern",
        help="Pattern representing the full path to RV files we want to correct",
    )
    psr_correct.add_argument(
        "-b, --save-bin",
        dest="save_bin",
        action="store_true",
        help="Whether we should also bin the files per day after correction and save separately",
    )
    psr_correct.add_argument(
        "-s, --skip-full",
        dest="save_full",
        action="store_false",
        help="Whether the full corrected file should be saved (useful to get binned only)",
    )
    psr_correct.add_argument(
        "-z",
        "--zp-version",
        dest="zp_version",
        type=str,
        default=None,
        help="Name of the ZPC version shown in corrected file names",
    )
    psr_correct.add_argument(
        "-e",
        "--ext",
        type=str,
        default="rdb",
        help="Extension of the files if rvpattern is a directory",
    )
    psr_correct.set_defaults(func=driver.correct)

    args = psr.parse_args()

    # Whatever we do, load config first and override relevant args
    if "config" not in args:
        raise ArgumentError(
            config_arg,
            "A configuration file must be specified after the command.\n"
            " Example: brav0 {command} config.yml",
        )
    config = ut.load_config(args.config)
    config = config_override(config, args)
    # Ensure output directory is a path
    config.out_dir = Path(config.out_dir)

    args.func(config)


if __name__ == "__main__":
    main()
