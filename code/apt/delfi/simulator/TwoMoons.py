import delfi.distribution as dd
import numpy as np
from delfi.simulator.BaseSimulator import BaseSimulator


def default_mapfunc(theta, p):
    ang = -np.pi / 4.0
    c = np.cos(ang)
    s = np.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return p + np.array([-np.abs(z0), z1])


class TwoMoons(BaseSimulator):
    def __init__(self, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0,
                 mapfunc=default_mapfunc,  # transforms noise dist.
                 seed=None):
        """Two Moons simulator

        Toy model that draws data from a mixture distribution with 2 components
        that are cresent shaped have mean theta and fixed noise.

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        noise_cov : list
            Covariance of noise on observations
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=2, seed=seed)
        self.mean_radius = mean_radius
        self.sd_radius = sd_radius
        self.baseoffset = baseoffset
        self.mapfunc = mapfunc

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        a = np.pi * (self.rng.rand() - 0.5)
        r = self.mean_radius + self.rng.randn() * self.sd_radius
        p = np.array([r * np.cos(a) + self.baseoffset, r * np.sin(a)])
        return {'data': self.mapfunc(param, p)}

    def gen_posterior_samples(self, obs=np.array([0.0, 0.0]), prior=None, n_samples=1):
        # works only when we use the default_mapfunc above

        # use opposite rotation as above
        ang = -np.pi / 4.0
        c = np.cos(-ang)
        s = np.sin(-ang)

        theta = np.zeros((n_samples, 2))
        for i in range(n_samples):
            p = self.gen_single(np.zeros(2))['data']
            q = np.zeros(2)
            q[0] = p[0] - obs[0]
            q[1] = obs[1] - p[1]

            if np.random.rand() < 0.5:
                q[0] = -q[0]

            theta[i, 0] = c * q[0] - s * q[1]
            theta[i, 1] = s * q[0] + c * q[1]

        return theta