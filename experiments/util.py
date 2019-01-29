
import numpy as np
import scipy
import pickle
import os

import delfi.generator as dg
import delfi.distribution as dd
from delfi.simulator import BaseSimulator
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from delfi.utils.progress import no_tqdm, progressbar

import snl.pdfs as pdfs
import snl.simulators.lotka_volterra as sim_lv
import snl.simulators.mg1 as sim_mg1
import snl.simulators.gaussian as sim_gauss

import snl.inference.diagnostics.two_sample as two_sample
import snl.util as util
from snl.util import math
from snl.inference.abc import SMC, calc_dist


# utility for running 'Gauss', Lotka-Volterra and M/G/1 experiments from snl package 

# rejection of bad simulations - Lotka-Volterra not guaranteed to behave well
# (rejection based on inspection of simulator code in G. Papamakarios' snl package)
def stubborn_defaultrej(x):
    if x is None:
        return False
    elif np.any([u is None for u in x]):
        return False
    elif np.any(np.isnan(x)):
        return False
    else:
        return True


def load_base_setup():

    setup_dict = {}

    # training schedule
    setup_dict['n_train']=1000
    setup_dict['n_rounds'] = 1 # changes with experiment (to be overwritten) !

    # fitting setup
    setup_dict['minibatch']=100
    setup_dict['epochs']=500

    # network setup
    setup_dict['n_hiddens']=[50,50]
    setup_dict['reg_lambda']=0.01

    # convenience
    setup_dict['pilot_samples']=1000
    setup_dict['verbose']=True
    setup_dict['prior_norm']=False

    # SNPE-C parameters
    setup_dict['n_null'] = setup_dict['minibatch'] - 1
    setup_dict['proposal'] = 'discrete'
    setup_dict['moo'] = 'resample'

    # MAF parameters
    setup_dict['mode']='random' # ordering of variables for MADEs
    setup_dict['n_mades'] = 5 # number of MADES
    setup_dict['act_fun'] = 'tanh'
    setup_dict['batch_norm'] = False # batch-normalization currently not supported
    setup_dict['svi']=False          # regularization of MAF parameters
    setup_dict['train_on_all'] = True

    return setup_dict


def save_results_byname(posteriors=None, exp_id=None, path=None, **save_vars):
    assert exp_id is not None
    if path is None:
        path = 'results/'

    dir = os.path.join(path, exp_id)
    if not os.path.exists(dir):
        os.makedirs(dir)

    for varname in save_vars.keys():
        fn = os.path.join(path, exp_id, varname)
        np.save(fn, save_vars[varname])

    if posteriors is not None:
        fn = os.path.join(dir, 'posteriors')
        with open(fn + '.pickle', 'wb') as f:
            pickle.dump(posteriors, f, pickle.HIGHEST_PROTOCOL)


def load_results_byname(exp_id=None, path=None):
    assert exp_id is not None
    if path is None:
        path = 'results/'

    dir = os.path.join(path, exp_id)
    load_vars = dict()
    varsfiles = [fn for fn in os.listdir(dir) if fn.endswith('.npy')]
    for fn in varsfiles:
        varname = os.path.splitext(fn)[0]
        load_vars[varname] = np.load(os.path.join(dir, fn)).tolist()

    posteriors = None,
    fn = os.path.join(dir, 'posteriors.pickle')
    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            posteriors = pickle.load(f)

    return posteriors, load_vars


def save_results(logs, tds, posteriors, setup_dict, exp_id, path=None):

    if path is None:
        path = 'results/'


    dir = os.path.join(path, exp_id)
    if not os.path.exists(dir):
        os.makedirs(dir)

    fn = os.path.join(path, exp_id, 'logs')
    np.save(fn, logs)

    fn = os.path.join(path, exp_id, 'tds')
    np.save(fn, tds)

    fn = os.path.join(path, exp_id, 'setup_dict')
    np.save(fn, setup_dict)

    fn = os.path.join(path, exp_id, 'posteriors')
    with open(fn + '.pickle', 'wb') as f:
        pickle.dump(posteriors, f, pickle.HIGHEST_PROTOCOL)


def load_results(exp_id, path=None):

    if path is None:
        path = 'results/'

    fn = os.path.join(path, exp_id, 'logs')
    logs = np.load(fn + '.npy')[()].tolist()

    fn = os.path.join(path, exp_id, 'tds')
    tds = np.load(fn + '.npy')[()].tolist()

    fn = os.path.join(path, exp_id, 'setup_dict')
    setup_dict = np.load(fn + '.npy')

    fn = os.path.join(path, exp_id, 'posteriors')
    with open(fn + '.pickle', 'rb') as f:
        posteriors = pickle.load(f)

    return logs, tds, posteriors, setup_dict

# Lotka-Volterra model


class StubbornGenerator(dg.Default):
    """ stubborn generator samples until it has n_samples simulation outcomes.
        Relevant for Lotka-Volterra and other simulators that not always return
        succesfull simulation outcomes.
    """

    def gen(self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True, recurse=False):
        params, stats = super().gen(n_samples,
                                    skip_feedback=False,  # never skip
                                    prior_mixin=prior_mixin,
                                    verbose=verbose)

        n_rem = n_samples - params.shape[0]

        if n_rem > 0:            
            params_rem, stats_rem = self.gen(n_rem,
                                            skip_feedback,
                                            prior_mixin,
                                            verbose, recurse=True)
            if params.shape[0] > 0:
                params = np.concatenate([params, params_rem], axis=0)
                stats = np.concatenate([stats, stats_rem], axis=0)
            else:
                params = params_rem
                stats = stats_rem

        assert params.shape[0] == stats.shape[0] == n_samples
        if verbose and not recurse:
            print('stubborn generator got {0} samples'.format(n_samples))
        return params, stats

    def _feedback_forward_model(self, data):
        for d in data:
            if d['data'] is None or np.isnan(d['data']).any():
                return 'discard'
        return 'accept'

    def _feedback_summary_stats(self, sum_stats):
        if np.isnan(sum_stats).any():
            return 'discard'
        return 'accept'


class StubbornGenerator_snl(dg.RejKernel):
    """ stubborn generator samples until it has n_samples simulation outcomes.
        Relevant for Lotka-Volterra and other simulators that not always return
        succesfull simulation outcomes.
    """

    def gen(self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True):

        params, stats = super().gen(n_samples,
                                    skip_feedback=False,  # never skip
                                    prior_mixin=prior_mixin,
                                    verbose=verbose)

        n_rem = n_samples - params.shape[0]

        if n_rem > 0:
            params_rem, stats_rem = self.gen(n_rem,
                                            skip_feedback,
                                            prior_mixin,
                                            verbose)
            if params.shape[0] > 0:
                params = np.concatenate([params, params_rem], axis=0)
                stats = np.concatenate([stats, stats_rem], axis=0)
            else:
                params = params_rem
                stats = stats_rem

        assert params.shape[0] == stats.shape[0] == n_samples

        return params, stats 

def init_g_lv(seed):
    """ initializes generator object for Lotka-Volterra experiment

    """
    prior = dd.Uniform(lower= [-5,-5,-5,-5], upper = [2,2,2,2])

    # model
    model_snl = sim_lv.Model()

    class Lotka_volterra(BaseSimulator):
        """Lotka Volterra simulator

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        seed : int or None
            If set, randomness is seeded
        """

        def gen_single(self, params):
            """ params = (predator births, predator deaths,
                          prey births, predator-prey interactions)

            """
            return model_snl.sim(params)

    model = Lotka_volterra(dim_param=4)

    # summary statistics
    summary = sim_lv.Stats()
    summary.n_summary = 9

    # generator
    g = StubbornGenerator_snl(prior=prior, model=model, summary=summary, 
                              rej=stubborn_defaultrej, seed=seed+41)

    return g


def load_gt_lv(generator=None):

    try:
        gt = np.load('results/gt_lv.npy', encoding='latin1')[()]
        whiten_params = np.load('results/whiten_params_lv.npy', encoding='latin1')[()]
        pars_true, obs_stats = np.array(gt['true_ps']), np.array(gt['obs_xs']).reshape(1,-1)

        obs_stats = (obs_stats.flatten() * whiten_params['stds']) + whiten_params['means']
        obs_stats = obs_stats.reshape(1,-1)

    except:
        pars_true = np.log([0.01, 0.5, 1.0, 0.01])  # taken from SNL paper
        obs = generator.model.gen_single(pars_true)  # should also recover
        obs_stats = generator.summary.calc([obs])    # xo from SNL paper !

        print('\n WARNING: could not load ground-truth data and parameters from disk! \n Sampling xo instead !')

    return pars_true, obs_stats


def load_setup_lv():

    setup_dict = load_base_setup()
    setup_dict['n_rounds'] = 10

    return setup_dict


def draw_sample_uniform_prior_52(post, n_samples, batch=None, patience=100):
    """ Convencience rejection sampler for posterior estimates for the 'Lotka-Volterra' model.
        Over many rounds, MAFs naively trained with discrete-proposal SNPE-C tend
        to lose a lot of mass outside of the prior bounds. 

        To still sample correctly from the posterior, the MAF has to be truncated, or 
        we need to implement rejection sampling. This function allows rejection sampling
        much faster than the serially-sampling RejKernel() generator object of delfi. 

    """
    
    batch = n_samples if batch is None else None
    n_drawn, samples, ct = 0, [], 0
    while n_drawn < n_samples:
        minibatch = post.gen(batch)
        idx = np.where(np.prod(np.abs(minibatch + 1.5)< 3.5,axis=1))[0]
        samples.append(minibatch[idx])
        n_drawn += idx.size
        
        ct += 1        
        if ct > patience :
            samples.append(np.nan * np.ones((n_samples-n_drawn,minibatch.shape[1])))
            break
    print('sampling, (itercount, n_drawn) = ', (ct,n_drawn))
    return np.concatenate(samples, axis=0)[:n_samples]


# M/G/1 model


class ShiftedUniform(dd.Uniform):

    def __init__(self, lower=0., upper=1., seed=None):
        """Shifted uniform distribution from SNL paper, M/G/1 model
        theta1 ~ Unif[0,10]
        theta2-theta1 ~ Unif[0,10]
        theta3 ~ Unif[0,1/3]

        Parameters
        ----------
        lower : list, or np.array, 1d
            Lower bound(s)
        upper : list, or np.array, 1d
            Upper bound(s)
        seed : int or None
            If provided, random number generator will be seeded
        """

        super().__init__(lower=lower, upper=upper, seed=seed)
        assert self.ndim == 3

    def gen(self, n_samples=1):

        params = super().gen(n_samples=n_samples)
        params[:,1] += params[:,0]
        return params

    def eval(self, x, ii=None, log=True):

        x = x.copy()
        x[:,1] -= x[:,0]
        return super().eval(x=x, ii=ii,log=log)


class GenMG1(dg.Default):
    """ Generator class that allows rejection-sampling for uniform prior
        with non-rectangular support

    """

    def draw_params(self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True):

        proposal = self.prior if self.proposal is None else self.proposal

        return draw_sample_shifted_uniform_prior(proposal,
                                                 n_samples,
                                                 batch=None,
                                                 patience=10000)


def init_g_mg1(seed):
    """ initializes generator object for M/G/1 experiment

    """

    # prior
    prior = ShiftedUniform(lower=[ 0, 0,  0  ],
                            upper=[10,10,1./3.],
                            seed=seed)

    # model
    model_snl = sim_mg1.Model()
    class MG1(BaseSimulator):
        """M/G/1 simulator

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        seed : int or None
            If set, randomness is seeded
        """

        def gen_single(self, params):
            """ params = (lower bound of server processing time,
                           upper bound of server processing time,
                          rate customer arrivals )

            """
            return model_snl.sim(params, rng=self.rng)

    model = MG1(dim_param=3, seed=seed)

    # summary statistics
    summary = sim_mg1.Stats()
    summary.n_summary = 5

    # generator
    g = GenMG1(prior=prior, model=model, summary=summary, seed=seed+41)

    return g


def load_gt_mg1(generator=None):

    try:
        # load ground-truth xo and true parameter values theta* from disc
        gt = np.load('results/gt_mg1.npy', encoding='latin1')[()]
        whiten_params = np.load('results/whiten_params_mg1.npy', encoding='latin1')[()]
        pars_true, obs_stats = np.array(gt['true_ps']), np.array(gt['obs_xs']).reshape(1,-1)

        # un-whiten xo with retrieved whitening params (SNPE-C will apply its own z-scoring, but xo needs to match the x_n)
        obs_stats = (obs_stats.flatten() / whiten_params['istds']).dot(whiten_params['U'].T) + whiten_params['means']
        obs_stats = obs_stats.reshape(1,-1)

    except:
        pars_true = np.array([1, 5, 0.2])  # taken from SNL paper
        obs = generator.model.gen_single(pars_true)  # should also recover
        obs_stats = generator.summary.calc([obs])    # xo from SNL paper !

        print('\n WARNING: could not load ground-truth data and parameters from disk! \n Sampling xo instead !')

    return pars_true, obs_stats


def load_setup_mg1():

    setup_dict = load_base_setup()
    setup_dict['n_rounds'] = 20

    return setup_dict


def draw_sample_shifted_uniform_prior(post, n_samples, batch=None, patience=10000):
    """ Convencience rejection sampler for posterior estimates for the 'Gaussian' model.
        Due to the many rounds, MAFs naively trained with discrete-proposal SNPE-C tend
        to lose a lot of mass outside of the prior bounds. 

        To still sample correctly from the posterior, the MAF has to be truncated, or 
        we need to implement rejection sampling. This function allows rejection sampling
        much faster than the serially-sampling RejKernel() generator object of delfi. 

    """
    
    batch = n_samples if batch is None else None
    n_drawn, samples, ct = 0, [], 0
    while n_drawn < n_samples:
        minibatch = post.gen(batch)

        minibatch_ = minibatch.copy()
        minibatch_[:,1] -= minibatch_[:,0]
        minibatch_[:,2] *= 30
        idx = np.where(np.prod(np.abs(minibatch_-5)<5.,axis=1))[0]

        samples.append(minibatch[idx])
        n_drawn += idx.size
        
        ct += 1        
        if ct > patience :
            samples.append(np.nan * np.ones((n_samples-n_drawn,minibatch.shape[1])))
            break
    print('sampling, (itercount, n_drawn) = ', (ct,n_drawn))
    return np.concatenate(samples, axis=0)[:n_samples]


# 'Gaussian' model


class GenGauss(dg.Default):
    """ Generator class that allows parallelized rejection-sampling for uniform prior

    """

    def draw_params(self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True):

        proposal = self.prior if self.proposal is None else self.proposal

        return draw_sample_uniform_prior_33(proposal,
                                            n_samples,
                                            batch=None,
                                            patience=10000)



def init_g_gauss(seed):
    prior = dd.Uniform(lower= [-3,-3,-3,-3,-3], upper = [3,3,3,3,3])

    # model
    model_snl = sim_gauss.Model()
    class Gaussian(BaseSimulator):
        """Gaussian simulator

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        seed : int or None
            If set, randomness is seeded
        """

        def gen_single(self, params):
            """ params = (m1, m2, std1, std2, arctanh(corr_scaling))

            """
            return model_snl.sim(params)

    model = Gaussian(dim_param=5)

    # summary statistics
    summary = sim_gauss.Stats()
    summary.n_summary = 8

    # generator
    g = GenGauss(prior=prior, model=model, summary=summary, seed=seed+41)
    return g


def load_setup_gauss():

    setup_dict = load_base_setup()
    setup_dict['n_rounds'] = 40
    setup_dict['upper'] =  3.01 # will trigger box-constraint 
    setup_dict['lower'] = -3.01 # conditional MAF to be used

    return setup_dict


def load_gt_gauss(generator=None):

    try:
        gt = np.load('results/gt_gauss.npy', encoding='latin1')[()]
        pars_true, obs_stats = np.array(gt['true_ps']), np.array(gt['obs_xs']).reshape(1,-1)

    except:
        pars_true = np.array([-0.7, 2.9, -1., -0.9, 0.6])  # taken from SNL paper
        obs = generator.model.gen_single(pars_true)  # should also recover
        obs_stats = generator.summary.calc([obs])    # xo from SNL paper !

        print('\n WARNING: could not load ground-truth data and parameters from disk! \n Sampling xo instead !')

    return pars_true, obs_stats


def draw_sample_uniform_prior_33(post, n_samples, batch=None, patience=10000):
    """ Convencience rejection sampler for posterior estimates for the 'Gaussian' model.
        Due to the many rounds, MAFs naively trained with discrete-proposal SNPE-C tend
        to lose a lot of mass outside of the prior bounds. 

        To still sample correctly from the posterior, the MAF has to be truncated, or 
        we need to implement rejection sampling. This function allows rejection sampling
        much faster than the serially-sampling RejKernel() generator object of delfi. 

    """
    
    batch = n_samples if batch is None else None
    n_drawn, samples, ct = 0, [], 0
    while n_drawn < n_samples:
        minibatch = post.gen(batch)
        idx = np.where(np.prod(np.abs(minibatch)<3.,axis=1))[0]
        samples.append(minibatch[idx])
        n_drawn += idx.size
        
        ct += 1        
        if ct > patience :
            samples.append(np.nan * np.ones((n_samples-n_drawn,minibatch.shape[1])))
            break
    print('sampling, (itercount, n_drawn) = ', (ct,n_drawn))
    return np.concatenate(samples, axis=0)[:n_samples]



# evualuating results, plotting 

def calc_all_mmds(samples_true, n_samples, posteriors, init_g, rej=True):
    """ only called for 'Gaussian' simulator """

    all_mmds = []
    ct = 0
    for proposal in posteriors:
        
        ct += 1
        
        print('\n round #' + str(ct) + '/' + str(len(posteriors)))
        print('- sampling')
        if rej:
            samples = draw_sample_uniform_prior_33(proposal, n_samples, patience=10000)
        else:
            samples = proposal.gen(n_samples)

        if np.any(np.isnan(samples)): # fail to sample n_sample times
            all_mmds.append(np.inf)
        else:            
            print('- computing MMD')
            scale = math.median_distance(samples_true)
            mmd = two_sample.sq_maximum_mean_discrepancy(samples, samples_true, scale=scale)
            if isinstance(mmd, np.ndarray):
                mmd = mmd.flatten()[0]
            all_mmds.append(mmd)
        

    return np.array(all_mmds).flatten()


def calc_err(pars_true, samples, weights=None, std=None):
    """
    Calculates error (neg log prob of truth) for a set of possibly weighted samples.
    Code from http://github.com/gpapamak/snl/blob/master/plot_results_lprob.py !
    """
    n_samples = samples.shape[0]

    std = n_samples ** (-1.0 / (len(pars_true) + 4)) if std is None else std
    std = std * np.diag(np.std(samples,axis=0))        

    return -pdfs.gaussian_kde(samples, weights, std).eval(pars_true)


def calc_all_lprob_errs(pars_true, n_samples, posteriors, init_g, rej=True, 
    fast_sampler=None, std=None):

    all_prop_errs = []
    for proposal in posteriors:
        if rej:
            if not fast_sampler is None:
                samples = fast_sampler(proposal, n_samples)
            else:
                g = init_g(seed=42)
                g.proposal = proposal
                samples = np.array(g.draw_params(n_samples))
        else:
            samples = proposal.gen(n_samples)

        if np.any(np.isnan(samples)): # fail to sample, see draw_sample_uniform_prior_33()
            all_prop_errs.append(np.inf)
        else:
            prop_err = calc_err(pars_true, samples, std=std)
            all_prop_errs.append(prop_err)

    return np.array(all_prop_errs)




# SMC-ABC algorithm with rejection sampling step to respect prior boundaries

class rejSMC(SMC):
    
    def sample_initial_population(self, obs_data, n_particles, eps, logger, rng):
        """
        Sample an initial population of n_particles, with tolerance eps.
        """

        ps = []
        n_sims = 0

        for i in range(n_particles):

            dist = float('inf')
            prop_ps = None

            while dist > eps:
                while True:
                    prop_ps = self.prior.gen(rng=rng)
                    try:
                        self.prior.eval(prop_ps, log=True) # delfi Uniform priors return
                        break                              # errors if evaluated outside support
                    except:
                        pass
                data = self.sim_model(prop_ps, rng=rng)
                dist = calc_dist(data, obs_data)
                n_sims += 1

            ps.append(prop_ps)

            logger.write('particle {0}\n'.format(i + 1))

        return np.array(ps), n_sims

    def sample_next_population(self, ps, log_weights, obs_data, eps, logger, rng):
        """
        Samples a new population of particles by perturbing an existing one. Uses a gaussian perturbation kernel.
        """

        n_particles, n_dim = ps.shape
        n_sims = 0
        weights = np.exp(log_weights)

        # calculate population covariance
        mean = np.mean(ps, axis=0)
        cov = 2.0 * (np.dot(ps.T, ps) / n_particles - np.outer(mean, mean))
        std = np.linalg.cholesky(cov)

        new_ps = np.empty_like(ps)
        new_log_weights = np.empty_like(log_weights)

        for i in range(n_particles):

            dist = float('inf')

            while dist > eps:
                while True:
                    idx = util.math.discrete_sample(weights, rng=rng)
                    new_ps[i] = ps[idx] + np.dot(std, rng.randn(n_dim))
                    try:
                        self.prior.eval(new_ps[i], log=True) # delfi Uniform priors return
                        break                                # errors if evaluated outside support
                    except:
                        pass                
                data = self.sim_model(new_ps[i], rng=rng)
                dist = calc_dist(data, obs_data)
                n_sims += 1

            # calculate unnormalized weights
            log_kernel = -0.5 * np.sum(scipy.linalg.solve_triangular(std, (new_ps[i] - ps).T, lower=True) ** 2, axis=0)
            new_log_weights[i] = self.prior.eval(new_ps[i], log=True) - scipy.misc.logsumexp(log_weights + log_kernel)

            logger.write('particle {0}\n'.format(i + 1))

        # normalize weights
        new_log_weights -= scipy.misc.logsumexp(new_log_weights)

        return new_ps, new_log_weights, n_sims     

class NoiseStats(BaseSummaryStats):
    
    def __init__(self, noise_source, n_signal = None, seed=None):
        """ Summary statistics instance that adds iid noise from a given noise
            source to simulator output and permutes dimensions with a fixed ordering
        """

        # distribution to generative independent noise
        self.noise_source = noise_source
        
        self.n_signal = n_signal              # original dimensionality
        self.n_noise = self.noise_source.ndim # noise dimensionality m   

        super().__init__(seed=seed)
        
        self.n_summary = n_signal + self.n_noise # total dimensionality d
       
        rng = np.random
        rng.seed(seed)
        self.idx = np.arange(self.n_summary)       # permutate signal and
        self.idx = rng.permutation(self.n_summary) # noise dimensions (once!)
        
    def calc(self, repetition_list):
                
        # get the number of samples contained
        n_reps = len(repetition_list)

        # get the size of the data inside a sample
        assert self.n_summary == repetition_list[0].size + self.n_noise

        # build a matrix of n_reps x n_summary
        data_matrix = np.zeros((n_reps, self.n_summary))
        noise_matrix = self.noise_source.gen(n_reps)
        for rep_idx, rep_val in enumerate(repetition_list):
            data_matrix[rep_idx, :] =  np.hstack( (rep_val, noise_matrix[rep_idx,:]) )

        return data_matrix if self.idx is None else data_matrix[:, self.idx]
