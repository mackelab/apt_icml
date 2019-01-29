from copy import deepcopy
import numpy as np
from delfi.neuralnet.Trainer import Trainer
from delfi.generator.BaseGenerator import BaseGenerator
from delfi.neuralnet.loss.lossfunc import snpeabc_loss_prior_as_proposal


class SNLprior(object):
    """Wrapper for delfi.distribution objects to act like SNL.pdf objects.
    Key differences: 
    1. SNL does not have Generator() objects, instead prior and (model+summary)
       are passed as functions that can be used to sample. For the prior, this
       means we need to provide a fully fledged gen() method with own seeding.
    2. SNL pdfs have the 'one_sample' feature, which returns (dim,)-shaped
       outputs when generating a single sample, rather than shape (1,dim).
       For the prior, this means we need to support 'one_sample' for gen() and
       eval()

    Parameters
    ----------
    delfi_prior : delfi.distributions object
        Prior as would be used for delfi.generator.
    """
    def __init__(self, delfi_prior):
        self.delfi_prior = delfi_prior

    def gen(self, n_samples=None, rng=None):
        """Method to generate samples

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        rng : random stream
            Optional method to set rng (used by SNL)

        Returns
        -------
        n_samples x self.ndim if n_samples>1, else (self.ndim,)
        """        
        if not rng is None:
            self.delfi_prior.rng = rng
        
        if n_samples is None: 
            one_sample = True
            n_samples = 1
        else: 
            one_sample = False
            
        samples = self.delfi_prior.gen(n_samples)
        
        return samples[0] if one_sample else samples

    def eval(self, x, ii=None, log=True):
        """Method to evaluate pdf

        Parameters
        ----------
        x : int or list or np.array
            Rows are inputs to evaluate at
        ii : list
            A list of indices specifying which marginal to evaluate.
            If None, the joint pdf is evaluated
        log : bool, defaulting to True
            If True, the log pdf is evaluated

        Returns
        -------
        array (if more than one datapoint x), else scalar
        """        
        x = np.asarray(x)
        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]
        return self.delfi_prior.eval(x=x, ii=ii, log=log)


class SNLmodel(BaseGenerator):
    """Wrapper for delfi.generator objects to act like SNL.sim_data().
    Key differences: 
    1. SNL does not have Generator() objects, instead prior and (model+summary)
       are passed as functions that can be used to sample. For the generator, 
       this means we need to separate the simulator model from the (proposal)
       prior, and to directly extract summary stats from the simulations.
    2. SNL pdf's have the 'one_sample' feature, which returns (dim,)-shaped
       ouptputs when generating a single sample, rather than shape (1,dim). 
       For the generator, this means we need to support 'one_sample' convention
       in both the input (ps) and output (stats). 

    Parameters
    ----------
    delfi_model : delfi.simulator object
        Simulator as would be used for delfi.simulator.
    summary: summary statistics object
        Summary stats object with .calc([x]) method
    """
    def __init__(self, delfi_model, summary):
        self.delfi_model = delfi_model
        self.summary = summary

    def gen(self, ps, rng=None, skip_feedback=False, minibatch=50):
        """Run forward model given parameters

        Parameters
        ----------
        ps: array
            Parameters to simulate from. (n_sampled, n_params) or (n_params,)
        rng : random stream
            Optional method to set rng (used by SNL)

        Returns
        -------
        stats : n_samples x n_summary, or n_summary array
            Summary statistics of data
        """

        if not rng is None:
            self.delfi_model.rng = rng

        # support one_sample coming from the (proposal prior):
        if ps.ndim>1:
            params = [ps[i] for i in range(ps.shape[0])]
        else:
            params = [ps]
                        
        # Run forward model for params (in batches)
        final_params = []
        final_stats = []  # list of summary stats
        for params_batch in self.iterate_minibatches(params, minibatch):

            # run forward model for all params, each n_reps times
            result = self.delfi_model.gen(params_batch, n_reps=1, pbar=None)

            stats, params = self.process_batch(params_batch, 
                                            result,skip_feedback=skip_feedback)
            final_params += params
            final_stats += stats

        # n_samples x n_reps x dim theta
        params = np.array(final_params)

        # n_samples x n_reps x dim summary stats
        stats = np.array(final_stats)
        stats = stats.squeeze() # already supports 'one_sample' !

        return stats


class MAFconditional(object):
    """Distribution-like object for masked autoregressive flows.

    Allows to sample and evaluate densities for parameters from the posterior
    estimate as required for SNPE(-C).
    """
    def __init__(self, model, obs_stats, makecopy=False, rng=np.random):
        from snl.ml.models.mafs import ConditionalMaskedAutoregressiveFlow
        assert isinstance(model, ConditionalMaskedAutoregressiveFlow)
        self.model = deepcopy(model) if makecopy else model
        assert obs_stats.ndim == 1
        self.obs_stats = obs_stats
        self.rng = rng

    def gen(self, n_samples, x=None):
        """Method to generate samples

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_samples : array
            Vector of summary stats (for conditional density param|stats )

        Returns
        -------
        n_samples x self.ndim
        """
        if not x is None:
            assert x.ndim==1
        return self.model.gen(
            x=self.obs_stats if x is None else x,
            n_samples=n_samples,
            rng=self.rng)

    def eval(self, x, log=True):
        """Method to evaluate pdf

        Parameters
        ----------
        x : int or list or np.array
            Rows are inputs to evaluate at
        log : bool, defaulting to True
            If True, the log pdf is evaluated

        Returns
        -------
        scalar
        """        
        assert x.ndim < 3
        return self.model.eval(
            xy=(self.obs_stats, x),
            log=log)


def normalize_cmaf(cmaf, f_accept, xs, n_samples=10, seed=None, val_frac=0.1,
                   minibatch=100, epochs=200, verbose=False, stop_on_nan=False):
    from snl.ml.models.mafs import ConditionalMaskedAutoregressiveFlow
    assert isinstance(cmaf, ConditionalMaskedAutoregressiveFlow)

    # first, sample from the existing MAF and apply our acceptance criterion
    xs = np.repeat(xs, n_samples, axis=0)
    thetas = np.full(xs.shape, np.nan)
    rng = np.random.randomstate(seed=seed)

    jj = 0  # index into final array of thetas
    for i, x in enumerate(xs):
        n_accepted = 0
        while n_accepted < n_samples:
            next_thetas = \
                cmaf.gen(x=x, n_samples=n_samples - n_accepted, rng=rng)
            for theta in next_thetas:
                if not f_accept(theta):
                    continue
                thetas[jj, :] = theta
                n_accepted += 1
                jj += 1

    # create a new MAF
    newrng = np.random.RandomState(seed=seed)
    cmaf_new = ConditionalMaskedAutoregressiveFlow(
        n_inputs=cmaf.n_inputs,
        n_outputs=cmaf.n_outputs,
        n_hiddens=cmaf.n_hiddens,
        act_fun=cmaf.act_fun,
        n_mades=cmaf.n_mades,
        batch_norm=cmaf.batch_norm,
        mode=cmaf.mode,
        input=None,  # is this ok?
        output=None,  # is this ok?
        rng=newrng,
        output_order=cmaf.output_order)  # hope this is ok?

    # train network by directly maximizing q(\theta | x)
    loss, trn_inputs = snpeabc_loss_prior_as_proposal(cmaf, svi=False)
    trn_data = (thetas, xs)

    t = Trainer(network=cmaf, loss=loss, trn_data=trn_data,
                trn_inputs=trn_inputs, seed=seed + 5)
    log = t.train(epochs=epochs, minibatch=minibatch,
                  verbose=verbose, stop_on_nan=stop_on_nan, val_frac=val_frac)
