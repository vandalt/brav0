from argparse import ArgumentParser, Namespace
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
    psr_parent.add_argument("config", type=str, help="Configuration file.")
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

    args = psr.parse_args()

    # Whatever we do, load config first and override relevant args
    config = ut.load_config(args.config)
    config = config_override(config, args)
    # Ensure output directory is a path
    config.out_dir = Path(config.out_dir)

    args.func(config)


if __name__ == "__main__":
    main()
