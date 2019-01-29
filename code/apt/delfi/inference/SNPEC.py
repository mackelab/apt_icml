import numpy as np
import theano
import delfi.distribution as dd
from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer, ActiveTrainer
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero
from delfi.neuralnet.loss.lossfunc import \
    (snpeabc_loss_prior_as_proposal, snpec_loss_gaussian_proposal,
     snpec_loss_MoG_proposal, snpec_loss_discrete_proposal)
from delfi.neuralnet.NeuralNet import NeuralNet, dtype

from delfi.utils.delfi2snl import MAFconditional
from delfi.utils.data import repnewax, combine_trn_datasets


class SNPEC(BaseInference):
    def __init__(self, generator, obs=None, prior_norm=False,
                 pilot_samples=100, reg_lambda=0.01, seed=None, verbose=True,
                 **kwargs):
        """SNPE-C
        Core idea is to parameterize the true posterior, and calculate the
        proposal posterior as needed on-the-fly. See work-in-progress latex
        file on overleaf.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        n_components : int
            Number of components in final round (PM's algorithm 2)
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)  # initializes network
        assert obs is not None, "snpec requires observed data"
        self.obs = np.asarray(obs)
        assert 0 < self.obs.ndim <= 2
        if self.obs.ndim == 1:
            self.obs = self.obs.reshape(1, -1)
        assert self.obs.shape[0] == 1

        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.exception_info = (None, None, None)
        self.trn_datasets, self.proposal_used = [], []

    def define_loss(self, N, round_cl=1, proposal='gaussian',
                    combined_loss=False):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        round_cl : int
            Round after which to start continual learning
        proposal : str
            Specifier for type of proposal used: continuous ('gaussian', 'mog')
            or 'discrete' proposals are implemented.
        combined_loss : bool
            Whether to include prior likelihood terms in addition to discrete
        """
        if proposal == 'prior':  # using prior as proposal
            loss, trn_inputs = snpeabc_loss_prior_as_proposal(self.network,
                svi=self.svi)
        elif proposal == 'gaussian':
            assert isinstance(self.generator.proposal, dd.Gaussian)
            loss, trn_inputs = snpec_loss_gaussian_proposal(self.network,
                self.generator.prior, svi=self.svi)
        elif proposal.lower() == 'mog':
            loss, trn_inputs = snpec_loss_MoG_proposal(self.network,
                self.generator.prior, svi=self.svi)
        elif proposal == 'discrete':
            loss, trn_inputs = snpec_loss_discrete_proposal(self.network,
                svi=self.svi, combined_loss=combined_loss)
        else:
            raise NotImplemented()

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.lprobs'] = self.network.lprobs
        self.observables['loss.raw_loss'] = loss

        if self.svi:
            if self.round <= round_cl:
                # weights close to zero-centered prior in the first round
                if self.reg_lambda > 0:
                    kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                           self.reg_lambda)
                else:
                    kl, imvs = 0, {}
            else:
                # weights close to those of previous round
                kl, imvs = svi_kl_init(self.network.mps, self.network.sps)

            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss, trn_inputs

    def run(self, n_rounds=1, proposal='gaussian', silent_fail=True, **kwargs):
        """Run algorithm
        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        proposal : str
            Specifier for type of proposal used: continuous ('gaussian', 'mog')
            or 'discrete' proposals are implemented.
        epochs : int
            Number of epochs used for neural network training
        minibatch : int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        round_cl : int
            Round after which to start continual learning
        stop_on_nan : bool
            If True, will halt if NaNs in the loss are encountered
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of distributions
            posterior after each round
        """
        logs = []
        trn_datasets = []
        posteriors = []

        for r in range(n_rounds):
            self.round += 1

            if silent_fail:
                try:
                    log, trn_data = self.run_round(proposal, **kwargs)
                except:
                    print('Round {0} failed'.format(self.round))
                    import sys
                    self.exception_info = sys.exc_info()
                    break
            else:
                log, trn_data = self.run_round(proposal, **kwargs)

            logs.append(log)
            trn_datasets.append(trn_data)
            posteriors.append(self.predict(self.obs))

        return logs, trn_datasets, posteriors

    def run_round(self, proposal='gaussian', **kwargs):

        if 'train_on_all' in kwargs.keys() and kwargs['train_on_all'] == True:
            kwargs['round_cl'] = np.inf
        self.proposal_used.append(proposal if self.round > 1 else 'prior')

        if proposal == 'prior' or self.round == 1:
            return self.run_prior(**kwargs)
        elif proposal == 'gaussian':
            return self.run_gaussian(**kwargs)
        elif proposal.lower() == 'mog':
            return self.run_MoG(**kwargs)
        elif proposal == 'discrete':
            return self.run_discrete(combined_loss=False, **kwargs)
        elif proposal == 'discrete_comb':
            return self.run_discrete(combined_loss=True, **kwargs)
        else:
            raise NotImplemented()

    def run_prior(self, n_train=100, epochs=100, minibatch=50, n_null=None, moo=None, train_on_all=False, round_cl=1,
                  stop_on_nan=False, monitor=None, verbose=False, **kwargs):

        # simulate data
        self.generator.proposal = self.generator.prior
        trn_data, n_train_round = self.gen(n_train)
        self.trn_datasets.append(trn_data)

        if train_on_all:
            prior_datasets = [d for i, d in enumerate(self.trn_datasets)
                              if self.proposal_used[i] == 'prior']
            trn_data = combine_trn_datasets(prior_datasets)
            n_train_round = trn_data[0].shape[0]

        # train network
        self.loss, trn_inputs = self.define_loss(N=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='prior')
        t = Trainer(self.network,
                    self.loss,
                    trn_data=trn_data, trn_inputs=trn_inputs,
                    seed=self.gen_newseed(),
                    monitor=self.monitor_dict_from_names(monitor),
                    **kwargs)
        log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch,
            verbose=verbose, stop_on_nan=stop_on_nan)

        return log, trn_data

    def run_gaussian(self, n_train=100, epochs=100, minibatch=50, n_null=None, moo=None,  train_on_all=False,
                     round_cl=1, stop_on_nan=False, monitor=None, verbose=False, **kwargs):

        # simulate data
        self.set_proposal(project_to_gaussian=True)
        prop = self.generator.proposal
        assert isinstance(prop, dd.Gaussian)
        trn_data, n_train_round = self.gen(n_train)

        # here we're just repeating the same fixed proposal, though we
        # could also introduce some variety if we wanted.
        prop_m = np.expand_dims(prop.m, 0).repeat(n_train_round, axis=0)
        prop_P = np.expand_dims(prop.P, 0).repeat(n_train_round, axis=0)
        trn_data = (*trn_data, prop_m, prop_P)
        self.trn_datasets.append(trn_data)

        if train_on_all:
            prev_datasets = []
            for i, d in enumerate(self.trn_datasets):
                if self.proposal_used[i] == 'gaussian':
                    prev_datasets.append(d)
                    continue
                elif self.proposal_used[i] != 'prior':
                    continue
                # prior samples. the Gauss loss will reduce to the prior loss
                if isinstance(self.generator.prior, dd.Gaussian):
                    prop_m = self.generator.prior.mean
                    P = self.generator.prior.P
                elif isinstance(self.generator.prior, dd.Uniform):
                    # model a uniform as an zero-precision Gaussian:
                    prop_m = np.zeros(self.generator.model.dim_param, dtype)
                    P = np.zeros(self.generator.model.dim_param, dtype)
                else:
                    continue
                prop_m = np.expand_dims(prop.m, 0).repeat(d[0].shape[0], axis=0)
                prop_P = np.expand_dims(prop.P, 0).repeat(d[0].shape[0], axis=0)
                prev_datasets.append((*d, prop_m, prop_P))

            trn_data = combine_trn_datasets(prev_datasets)
            n_train_round = trn_data[0].shape[0]

        # train network
        self.loss, trn_inputs = self.define_loss(N=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='gaussian')
        t = Trainer(self.network,
                    self.loss,
                    trn_data=trn_data, trn_inputs=trn_inputs,
                    seed=self.gen_newseed(),
                    monitor=self.monitor_dict_from_names(monitor),
                    **kwargs)

        log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch,
                      verbose=verbose, stop_on_nan=stop_on_nan)

        return log, trn_data


    def run_MoG(self, n_train=100, epochs=100, minibatch=50,
                n_null=None, moo=None, train_on_all=False, round_cl=1,
                stop_on_nan=False, monitor=None, verbose=False, **kwargs):
        assert not train_on_all, "train_on_all is not yet implemented for MoG "\
                "proposals"

        # simulate data
        self.set_proposal(project_to_gaussian=False)
        prop = self.generator.proposal
        assert isinstance(prop, dd.MoG)
        trn_data, n_train_round = self.gen(n_train)

        # here we're just repeating the same fixed proposal, though we
        # could also introduce some variety if we wanted.
        nc = prop.n_components
        prop_Pms = repnewax(np.stack([x.Pm for x in prop.xs], axis=0),
                            n_train_round)
        prop_Ps = repnewax(np.stack([x.P for x in prop.xs], axis=0),
                           n_train_round)
        prop_ldetPs = repnewax(np.stack([x.logdetP for x in prop.xs], axis=0),
                               n_train_round)
        prop_las = repnewax(np.log(prop.a), n_train_round)
        prop_QFs = \
            repnewax(np.stack([np.sum(x.Pm * x.m) for x in prop.xs], axis=0),
                     n_train_round)

        trn_data += (prop_Pms, prop_Ps, prop_ldetPs, prop_las,
                     prop_QFs)
        trn_data = tuple(trn_data)

        self.loss, trn_inputs = self.define_loss(N=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='mog')
        t = Trainer(self.network,
                    self.loss,
                    trn_data=trn_data, trn_inputs=trn_inputs,
                    seed=self.gen_newseed(),
                    monitor=self.monitor_dict_from_names(monitor),
                    **kwargs)

        log = t.train(epochs=epochs, minibatch=minibatch,
                      verbose=verbose, stop_on_nan=stop_on_nan)

        return log, trn_data

    def run_discrete(self, n_train=100, epochs=100, minibatch=50,
        n_null=10, moo='resample', train_on_all=False, combined_loss=False,
        round_cl=1, stop_on_nan=False, monitor=None, verbose=False, **kwargs):

        assert minibatch > 1, "minimum minibatch size is 2 for discrete proposals"
        if n_null is None:
            n_null = minibatch - 1 if theano.config.device.startswith('cuda') else np.minimum(minibatch - 1, 9)
        assert n_null < minibatch, "To small a minibatch size for this many nulls"
        # simulate data
        self.set_proposal()
        trn_data, n_train_round = self.gen(n_train)
        self.trn_datasets.append(trn_data)  # don't store prior_masks

        if train_on_all:
            trn_data = combine_trn_datasets(self.trn_datasets, max_inputs=2)
            if combined_loss:
                prior_masks = \
                    [np.ones(td[0].shape[0], dtype) * (pu == 'prior')
                     for td, pu in zip(self.trn_datasets, self.proposal_used)]
                trn_data = (*trn_data, np.concatenate(prior_masks))
            n_train_round = trn_data[0].shape[0]

        # train network
        self.loss, trn_inputs = self.define_loss(N=n_train_round,
                                                 round_cl=round_cl,
                                                 proposal='discrete',
                                                 combined_loss=combined_loss)

        t = ActiveTrainer(self.network,
                          self.loss,
                          trn_data=trn_data, trn_inputs=trn_inputs,
                          seed=self.gen_newseed(),
                          monitor=self.monitor_dict_from_names(monitor),
                          generator=self.generator,
                          n_null=n_null,
                          moo=moo,
                          obs=(self.obs - self.stats_mean) / self.stats_std,
                          **kwargs)

        log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch,
                      stop_on_nan=stop_on_nan, verbose=verbose,
                      strict_batch_size=True)

        return log, trn_data


    def set_proposal(self, project_to_gaussian=False):
        # posterior estimate becomes new proposal prior
        if self.round == 0 :
            return None

        if isinstance(self.network, NeuralNet):
            posterior = self.predict(self.obs)

            if project_to_gaussian:
                posterior = posterior.project_to_gaussian()

            self.generator.proposal = posterior

        else:
            from snl.ml.models.mafs import ConditionalMaskedAutoregressiveFlow
            assert isinstance(self.network, ConditionalMaskedAutoregressiveFlow)

            self.generator.proposal = MAFconditional(model=self.network,
                obs_stats=((self.obs - self.stats_mean) / self.stats_std).flatten(),
                makecopy=True, rng=self.rng)


    def gen(self, n_train, project_to_gaussian=False, **kwargs):
        """Generate from generator and z-transform

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_reps : int
            Number of repeats per parameter
        verbose : None or bool or str
            If None is passed, will default to self.verbose
        project_to_gaussian: bool
            Whether to always return Gaussian objects (instead of MoG)
        """
        verbose = '(round {}) '.format(self.round) if self.verbose else False
        n_train_round = self.n_train_round(n_train)

        trn_data = super().gen(n_train_round, verbose=verbose, **kwargs)
        n_train_round = trn_data[0].shape[0]  # may have decreased (rejection)

        return trn_data, n_train_round


    def n_train_round(self, n_train):
        # number of training examples for this round
        if type(n_train) == list:
            try:
                n_train_round = n_train[self.round-1]
            except:
                n_train_round = n_train[-1]
        else:
            n_train_round = n_train

        return n_train_round

    def epochs_round(self, epochs):
        # number of training examples for this round
        if type(epochs) == list:
            try:
                epochs_round = epochs[self.round-1]
            except:
                epochs_round = epochs[-1]
        else:
            epochs_round = epochs

        return epochs_round

    def predict(self, x, deterministic=True):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        x = np.atleast_2d(x)
        try:
            if isinstance(self.network, NeuralNet):
                return super().predict(x)
            else:
                obz = ((x - self.stats_mean)/self.stats_std).flatten()
                return MAFconditional(
                    model=self.network,
                    obs_stats=obz,
                    makecopy=True,
                    rng=self.rng)
        except:
            print('Posterior inference failed')
            import sys
            self.exception_info = sys.exc_info()
            return None


    def reinit_network(self):
        """Reinitializes the network instance (re-setting the weights!)
        """
        keys = self.kwargs.keys()
        if 'n_mades' in keys:

            assert 'batch_norm' in keys and not self.kwargs['batch_norm']
            self.svi = False

            # some convention adjustments across code packages (model inputs)
            kwargs = self.kwargs.copy()
            kwargs['rng'] = np.random.RandomState(seed=kwargs['seed'])
            assert len(kwargs['n_inputs']) == 1, 'only vector-shaped inputs!'
            kwargs['n_inputs'] = kwargs['n_inputs'][0]
            for mdn_kwarg in ['seed', 'svi', 'n_rnn', 'n_inputs_rnn']:
                if mdn_kwarg in kwargs.keys():
                    kwargs.pop(mdn_kwarg)

            if 'n_hiddens' not in kwargs.keys():
                kwargs['n_hiddens'] = [50, 50]

            if 'upper' in keys:
                assert 'lower' in keys
                from delfi.utils.BoxConstraintConditionalAutoregressiveFlow import BoxConstraintConditionalAutoregressiveFlow
                self.network = BoxConstraintConditionalAutoregressiveFlow(**kwargs)
            else:
                from snl.ml.models.mafs import ConditionalMaskedAutoregressiveFlow
                self.network = ConditionalMaskedAutoregressiveFlow(**kwargs)

            # some convention adjustments across code packages (output model)
            self.network.aps = self.network.parms # list of model parameters
            self.network.lprobs = self.network.L  # model log-likelihood

        else:

            self.network = NeuralNet(**self.kwargs)
            self.svi = self.network.svi

        """update self.kwargs['seed'] so that reinitializing the network gives
        a different result each time unless we reseed the inference method"""
        self.kwargs['seed'] = self.gen_newseed()

        self.norm_init()
