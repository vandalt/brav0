# brav0

_brav0_ (**B**ayesian **Ra**dial **V**elocity **0**-point correction) is a tool
to correct zero-point variations in radial velocity (RV) timeseries. This means
_brav0_ takes several datasets (usually from the same instrument) and models
variations that are common to all datasets.

## Installation
_brav0_ can be installed with pip: `python -m pip install brav0`.

To use the development version of _brav0_, clone the repository and install it:
```shell
git clone https://github.com/vandalt/brav0.git
cd brav0
python -m pip install -U -e ".[dev]"
```

_Note: In both cases, as of release 0.1, the development pace will probably be
relatively fast for a while so users should update often, either by pulling
from the upstream Github repository or by upgrading with `python -m pip install
-U brav0`.

## Using _brav0_
_brav0_ is accessible as a command line script or as a Python library. The
script requires a configuration file. There are example config files as well as
a notebook using the API in the `examples` directory.

### brav0 CLI
The CLI is the main way to use _brav0_. It does not (yet) provide an command to
run everything at once. The main ZP correction steps are instead separated in
various commands.

First, we run `source` to load all the input individual data and merge it in a
single pandas dataframe.
```
brav0 source config.yml
```
This produces a `raw.csv` file in the output directory, indexed by original
file name.

Then, we can preprocess the data by doing a series cleanups and by re-formatting
the dataframe (e.g. index with object names).
```
brav0 preprocess config.yml
```
This produces the `processed.csv` file the `raw_plots` directory with timeseries
and periodogram plots before PP, and the `pp_plots` directory with plots after
PP.

Once the data is ready, we can remove known planets. Currently, the only way to
do this in `brav0` is to use the [NASA explanet archive](https://exoplanetarchive.ipac.caltech.edu/)
to remove known planets. It performs Monte-Carlo error propagation and removes
"non-controversial" planets only (as defined by the archive).
```
brav0 remove-planets config.yml
```
The resulting dataset is stored in `no_planets.csv` with corresponding plots in
`no_planet`.

After removing known planets, we can fit the Zero-point model joinlty to all
data. The config file specifies if we do MCMC, MAP optimization, or just use a
fixed model (recommended only when all parameters have deterministic values).
Here is an example where we fit a GP with a Matern 3/2 kernel:
```
brav0 model config.yml Matern32
```
This produces the model curve and the optimization or sampling results in a
directory with the model name (or other subdirectory when using the `-o`
option).

Finally, we can generate summary information and plots about a given ZP model:
```
brav0 summary config.yml /path/to/model/dir
```
This will save plots in the model directory.

## Why brav0 ?
Fitting RV zero-points can be done with relatively simple tools. _brav0_ was
originally written to explore the use of Gaussian processes to model RV
zero-points. When fitting a GP along with parameters for each standard
(calibration) star, the number of parameter can be high, such that sampling the
posterior distribution efficiently is challenging. _brav0_ uses PyMC3 to perform
gradient-based inference (other backends are not excluded, contributions are
welcome!). By using `exoplanet` and `celerite2`, _brav0_ enables efficient
inference to derive a zero-point correction error estimates.
