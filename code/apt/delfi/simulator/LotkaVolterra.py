import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats

import snl.simulators.markov_jump_processes as mjp


class LotkaVolterra(BaseSimulator):
    def __init__(self, initial_state=(50, 100), dt=0.2, duration=30.0, include_init_state=True,
                 max_n_steps=10000, noise_sd=0.0, seed=None):
        """Gauss simulator

        Toy model that draws data from a distribution centered on theta with
        fixed noise.

        Parameters
        ----------
        initial state : tuple of ints
            Initial population values
        dt : float
            Time step between observations
        duration : float
            Simulation length
        max_n_steps: int
            Max. number of allowed "reactions"
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=4, seed=seed)
        self.initial_state = np.asarray(initial_state)
        self.dt = dt
        self.duration = duration
        self.max_n_steps = max_n_steps
        self.mjpobj = mjp.LotkaVolterra(self.initial_state, None)
        self.include_init_state = include_init_state
        self.noise_sd = noise_sd

    @copy_ancestor_docstring
    def gen_single(self, params):
        # See BaseSimulator for docstring
        params = np.asarray(params).reshape(-1)
        params = np.exp(params)
        assert params.ndim == 1
        assert params.shape[0] == self.dim_param

        while True:
            try:
                self.mjpobj.reset(self.initial_state.copy(), params)
                states = self.mjpobj.sim_time(self.dt, self.duration,
                                              max_n_steps=self.max_n_steps,
                                              rng=self.rng,
                                              include_init_state=self.include_init_state)
                data = states.flatten()
                data += self.rng.randn(data.size) * self.noise_sd
                return {'data': data}
            except mjp.SimTooLongException:
                return {'data': None}


class LotkaVolterraStats(BaseSummaryStats):
    """10 summary stats for the Lotka-Volterra simulator, as defined in the SNL
    paper.
    """
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.n_summary = 9

    def calc(self, repetition_list):
        # get the number of samples contained
        n_reps = len(repetition_list)

        data_matrix = np.full((n_reps, self.n_summary), np.nan)
        for i, datum in enumerate(repetition_list):
            if datum['data'] is not None:
                data_matrix[i, :] = self._calc_one_datapoint(datum['data'])
        return data_matrix

    def _calc_one_datapoint(self, datum):
        # copied from /snl/simulators/lotka_volterra.py

        xy = np.reshape(datum, [-1, 2])
        x, y = xy[:, 0], xy[:, 1]
        n = xy.shape[0]

        # means
        mx = np.mean(x)
        my = np.mean(y)

        # variances
        s2x = np.var(x, ddof=1)
        s2y = np.var(y, ddof=1)

        # standardize
        x = (x - mx) / np.sqrt(s2x)
        y = (y - my) / np.sqrt(s2y)

        # auto correlation coefficient
        acx = []
        acy = []
        for lag in [1, 2]:
            acx.append(np.dot(x[:-lag], x[lag:]) / (n - 1))
            acy.append(np.dot(y[:-lag], y[lag:]) / (n - 1))

        # cross correlation coefficient
        ccxy = np.dot(x, y) / (n - 1)

        # normalize stats
        xs = np.array(
            [mx, my, np.log(s2x + 1.0), np.log(s2y + 1.0)] + acx + acy + [ccxy])

        return xs
