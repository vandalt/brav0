from typing import Optional, Union

import aesara_theano_fallback.tensor as tt
import numpy as np
# import numpy as np
import orbits.model as om
import pymc3 as pm
import pymc3_ext as pmx
import xarray as xr
from arviz.data.inference_data import InferenceData
from celerite2.theano import GaussianProcess
from orbits.kernel import CELERITE_KERNELS, KERNELS, PYMC3_KERNELS
# from celerite2.theano import GaussianProcess
# from orbits.kernel import CELERITE_KERNELS, KERNELS, PYMC3_KERNELS
from pandas.core.frame import DataFrame
from pymc3 import Model

import brav0.rolling as br


class ZeroPointModel(Model):
    def __init__(
        self,
        data: DataFrame,
        model_parameters: Optional[dict[str, dict]] = None,
        time_label: str = "rjd",
        vrad_label: str = "vrad",
        svrad_label: str = "svrad",
        obj_label: str = "OBJECT",
        name: str = "",
        model: Optional[Model] = None,
    ):
        """
        Base class for all zero-point correction model. This class defines
        common data attribute and methods other ZP models can use. It is
        a PyMC3 model so it can be accessed via a `with` context.

        :param data: DataFrame with at least time, RV and RV error columns
        :type data: DataFrame
        :param time_label: Label of the time column, defaults to "rjd"
        :type time_label: str, optional
        :param vrad_label: Label of the RV column, defaults to "vrad"
        :type vrad_label: str, optional
        :param svrad_label: RV Error column, defaults to "svrad"
        :type svrad_label: str, optional
        :param obj_label: Label of the column with object names,
                          defaults to "OBJECT"
        :type obj_label: str, optional
        :param name: PyMC3 model name, defaults to ""
        :type name: str, optional
        :param model: PyMC3 parent model, defaults to None
        :type model: Optional[Model], optional
        """

        super().__init__(name=name, model=model)

        self.time_lab = time_label
        self.vrad_lab = vrad_label
        self.svrad_lab = svrad_label
        self.obj_label = obj_label

        self.data = data.sort_values(self.time_lab)

        self.t = self.data[self.time_lab].values
        self.vrad = self.data[self.vrad_lab].values
        self.svrad = self.data[self.svrad_lab].values

        # This data container is required for "DataNormal" priors
        # NOTE: This may conflict with self.vrad above, in which case the
        # latter will be accessible with dot and the Data will be accessible
        # as a key, both from the model object
        pm.Data(self.vrad_lab, self.vrad)
        pm.Data(self.svrad_lab, self.svrad)

        # All zero-point models may have per-system sub-models, so create here
        self.obj_list = list(
            self.data.index.get_level_values(self.obj_label).unique()
        )

        if model_parameters is not None:
            resid_rv = tt.zeros(len(self.t)) + self.vrad
            diag_rv = tt.zeros(len(self.t)) + self.svrad ** 2
            for objname, odata in self.data.groupby(self.obj_label):

                # Get mask for when we'll subtract stuff from the full dataset
                obj_mask = (
                    self.data.index.get_level_values(self.obj_label) == objname
                )

                if objname in model_parameters:
                    obj_key = objname
                else:
                    obj_key = "general_object_parameters"

                obj_time = odata[self.time_lab].values
                obj_vrad = odata[self.vrad_lab].values
                obj_svrad = odata[self.svrad_lab].values

                obj_model = om.RVModel(
                    obj_time,
                    obj_vrad,
                    obj_svrad,
                    0,
                    params=model_parameters[obj_key],
                    name=objname,
                    model=self,
                )

                obj_mean = obj_model.named_vars[objname + "_gamma"]
                obj_jitter = obj_model.named_vars[objname + "_wn"]
                resid_rv -= obj_mean * obj_mask
                diag_rv += obj_jitter ** 2 * obj_mask

            self.sys_resid = resid_rv
            self.sys_diag = diag_rv
        else:
            self.sys_resid = self.vrad
            self.sys_diag = self.svrad ** 2

    def optimize(self, return_xarray: bool = False) -> Union[dict, xr.Dataset]:
        with self:
            map_soln = pmx.optimize()

        return xr.Dataset(map_soln) if return_xarray else map_soln

    def sample(
        self,
        tune: int = 1000,
        draws: int = 1000,
        start: Optional[dict] = None,
        cores: int = 4,
        chains: int = 4,
        target_accept: float = 0.95,
    ) -> InferenceData:
        with self:
            trace = pmx.sample(
                tune=tune,
                draws=draws,
                start=start,
                cores=cores,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
            )
        return trace


class RollingModel(ZeroPointModel):
    def __init__(
        self,
        data: DataFrame,
        window: int,
        center: bool = True,
        method: str = "mean",
        time_label: str = "rjd",
        vrad_label: str = "vrad",
        svrad_label: str = "svrad",
        obj_label: str = "OBJECT",
    ):
        """
        Rolling mean or median model to calculate the Zero-point curve.

        :param data: Full dataset use to calculate the ZP
        :type data: DataFrame
        :param window: Width of the window for rolling calculations (in days)
        :type window: int
        :param center: Wether the window should be centered on each data point,
                       defaults to True
        :type center: bool, optional
        :param method: Method to use ("mean" or "median"), defaults to "mean"
        :type method: str, optional
        :param time_label: Label for time column, defaults to "rjd"
        :type time_label: str, optional
        :param vrad_label: Label for RV column, defaults to "vrad"
        :type vrad_label: str, optional
        :param svrad_label: Label for RV Error column, defaults to "svrad"
        :type svrad_label: str, optional
        :param obj_label: Label of the column with object names,
                          defaults to "OBJECT"
        :type obj_label: str, optional
        """

        self._KNOWN_METHODS = ["mean", "median", "wmean", "wmedian"]

        super().__init__(
            data,
            model_parameters=None,
            time_label=time_label,
            vrad_label=vrad_label,
            svrad_label=svrad_label,
            obj_label=obj_label,
        )

        self.window = window / 2
        self.center = center
        if method in self._KNOWN_METHODS:
            self.method = method
        else:
            raise ValueError(
                f"Unsupported method, please use one of {self._KNOWN_METHODS}"
            )

        # Data formatting to have a dataset compatible with ZP calculation
        keep_labels = [
            self.time_lab,
            self.vrad_lab,
            self.svrad_lab,
        ]
        self.data = self.data[keep_labels]

    def get_zp_curve(self) -> DataFrame:

        t = self.data[self.time_lab].copy().values
        vrad = self.data[self.vrad_lab].copy().values
        w = self.data[self.svrad_lab].copy().values ** -2

        if self.method == "median":
            w = np.ones_like(vrad)
            rval, rerr = br.apply_moving(vrad, self.window, t, w, br.roll_med)
        elif self.method == "mean":
            w = np.ones_like(vrad)
            rval, rerr = br.apply_moving(vrad, self.window, t, w, br.roll_mean)
        elif self.method == "wmedian":
            rval, rerr = br.apply_moving(vrad, self.window, t, w, br.roll_med)
        elif self.method == "wmean":
            rval, rerr = br.apply_moving(vrad, self.window, t, w, br.roll_mean)
        else:
            raise ValueError(
                f"Unsupported method: use one of {self._KNOWN_METHODS}."
                " This should not happen: methods are checked at"
                " initialization. Please report this issue on Github."
            )

        zp_curve = DataFrame(
            {self.time_lab: t, self.vrad_lab: rval, self.svrad_lab: rerr}
        )

        return zp_curve


class RollingModelPymc(ZeroPointModel):
    def __init__(
        self,
        data: DataFrame,
        window: int,
        model_parameters: dict[str, dict],
        center: bool = True,
        method: str = "mean",
        time_label: str = "rjd",
        vrad_label: str = "vrad",
        svrad_label: str = "svrad",
        obj_label: str = "OBJECT",
    ):
        """
        Rolling mean or median model to calculate the Zero-point curve.

        :param data: Full dataset use to calculate the ZP
        :type data: DataFrame
        :param window: Width of the window for rolling calculations (in days)
        :type window: int
        :param center: Wether the window should be centered on each data point,
                       defaults to True
        :type center: bool, optional
        :param method: Method to use ("mean" or "median"), defaults to "mean"
        :type method: str, optional
        :param time_label: Label for time column, defaults to "rjd"
        :type time_label: str, optional
        :param vrad_label: Label for RV column, defaults to "vrad"
        :type vrad_label: str, optional
        :param svrad_label: Label for RV Error column, defaults to "svrad"
        :type svrad_label: str, optional
        :param obj_label: Label of the column with object names,
                          defaults to "OBJECT"
        :type obj_label: str, optional
        """

        self._KNOWN_METHODS = ["mean", "median", "wmean", "wmedian"]

        super().__init__(
            data,
            model_parameters=model_parameters,
            time_label=time_label,
            vrad_label=vrad_label,
            svrad_label=svrad_label,
            obj_label=obj_label,
        )

        self.window = window / 2
        self.center = center
        self.tpred = self.data[self.time_lab].values.copy()
        if method in self._KNOWN_METHODS:
            self.method = method
        else:
            raise ValueError(
                f"Unsupported method, please use one of {self._KNOWN_METHODS}"
            )

        # Data formatting to have a dataset compatible with ZP calculation
        keep_labels = [
            self.time_lab,
            self.vrad_lab,
            self.svrad_lab,
        ]
        self.data = self.data[keep_labels]

        t = self.data[self.time_lab].copy().values
        vrad = self.sys_resid
        # w = self.data[self.svrad_lab].copy().values ** -2
        w = self.sys_diag ** -1
        if self.method == "median":
            w = np.ones_like(vrad)
            rval, rerr = br.apply_moving(
                vrad, self.window, t, w, br.roll_med, use_np=False
            )
        elif self.method == "mean":
            w = np.ones_like(vrad)
            rval, rerr = br.apply_moving(
                vrad, self.window, t, w, br.roll_mean, use_np=False
            )
        elif self.method == "wmedian":
            rval, rerr = br.apply_moving(
                vrad, self.window, t, w, br.roll_med, use_np=False
            )
        elif self.method == "wmean":
            rval, rerr = br.apply_moving(
                vrad, self.window, t, w, br.roll_mean, use_np=False
            )
        else:
            raise ValueError(
                f"Unsupported method: use one of {self._KNOWN_METHODS}."
                " This should not happen: methods are checked at"
                " initialization. Please report this issue on Github."
            )

        pm.Normal(
            "obs",
            mu=rval,
            sd=tt.sqrt(self.sys_diag),
            observed=self.sys_resid,
        )

        pm.Deterministic("pred", rval)
        pm.Deterministic("pred_std", rerr)


class GPModel(ZeroPointModel, Model):
    def __init__(
        self,
        data: DataFrame,
        kernel: str,
        model_parameters: dict[str, dict],
        time_label: str = "rjd",
        vrad_label: str = "vrad",
        svrad_label: str = "svrad",
        obj_label: str = "OBJECT",
        name: str = "",
        model: Optional[Model] = None,
        quiet_celerite: bool = False,
        tpred: Optional[np.ndarray] = None,
        tpred_num: Optional[int] = None,
    ):
        # First, we set-up the general ZP Model attributes
        # NOTE: ZeroPointModel will pass pymc3 model kwargs to pm.Model parent
        super().__init__(
            data,
            model_parameters=model_parameters,
            time_label=time_label,
            vrad_label=vrad_label,
            svrad_label=svrad_label,
            obj_label=obj_label,
            name=name,
            model=model,
        )

        # Time values for predicitive curve
        if tpred is None and tpred_num is None:
            # tpred = np.linspace(self.t.min(), self.t.max(), num=1000)
            tpred = np.linspace(self.t.min(), self.t.max(), num=400)
        elif tpred is None:
            tpred = np.linspace(self.t.min(), self.t.max(), num=tpred_num)
        elif tpred_num is not None:
            raise TypeError(
                "Only one of tpred and tpred_num should be provided."
            )
        # Include the measurement times, in case used for residuals
        self.tpred = np.unique(np.append(tpred, self.t))

        if kernel is None:
            pm.Normal(
                "obs",
                mu=0.0,
                sd=tt.sqrt(self.sys_diag),
                observed=self.sys_resid,
            )
            # pm.Normal("obs", mu=0.0, sd=np.sqrt(diag_rv), observed=resid_rv)
        else:
            # We first set-up the joint GP (ZP model) parameters
            self.gpmodel = om.GPModel(kernel, params=model_parameters["gp"])

            if kernel in CELERITE_KERNELS:
                self.gp = GaussianProcess(
                    self.gpmodel.kernel,
                    t=self.t,
                    diag=self.sys_diag,
                    # diag=diag_rv,
                    quiet=quiet_celerite,
                )
                # Compute GP marginalized log-like in pymc3 model
                self.gp.marginal("obs", observed=self.sys_resid)
                # self.gp.marginal("obs", observed=resid_rv)

                # Save the GP prediction in our model.
                pred, pred_var = self.gp.predict(
                    self.sys_resid,
                    t=self.tpred,
                    return_var=True
                    # resid_rv, t=self.tpred, return_var=True
                )
                pm.Deterministic("pred", pred)
                pm.Deterministic("pred_std", np.sqrt(pred_var))
            elif kernel in PYMC3_KERNELS:
                self.gp = pm.gp.Marginal(cov_func=self.gpmodel.kernel)
                self.gp.marginal_likelihood(
                    "obs",
                    self.t[:, None],
                    self.sys_resid,
                    noise=tt.sqrt(self.sys_diag)
                    # "obs", self.t[:, None], resid_rv, noise=np.sqrt(diag_rv)
                )
                pred, pred_var = self.gp.predictt(
                    self.tpred[:, None], diag=True
                )
                pm.Deterministic("pred", pred)
                pm.Deterministic("pred_std", np.sqrt(pred_var))
            else:
                raise ValueError(
                    f"gp_kernel must be None, or one of {list(KERNELS)}"
                )
