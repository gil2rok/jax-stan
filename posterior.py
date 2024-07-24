import bridgestan as bs
import numpy as np


class Posterior:
    def __init__(self, seed, model_path, data_path):
        self.bsmodel = bs.StanModel(
            model_lib=model_path,
            data=data_path,
            seed=seed,
            make_args=["STAN_THREADS=True", "TBB_CXX_TYPE=gcc"],
        )
        self.dimensions = self.bsmodel.param_unc_num()

    def logdensity(self, x):
        return self.bsmodel.log_density(x)

    def logdensity_and_gradient(self, x):
        log_density, gradient = self.bsmodel.log_density_gradient(x)

        if np.isnan(log_density) or np.isnan(gradient).any():
            raise ValueError(f"NaN values in log density or gradient at {x}")

        return log_density, gradient

    def unconstrain(self, x):
        return self.bsmodel.param_unconstrain(x)

    def constrain(self, x):
        return self.bsmodel.param_constrain(x)

    @property
    def dims(self):
        return self.dimensions
