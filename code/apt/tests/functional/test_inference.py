import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
import delfi.summarystats as ds
import numpy as np

from delfi.simulator.Gauss import Gauss


def simplegaussprod(m1, S1, m2, S2):
    m1 = m1.squeeze()
    m2 = m2.squeeze()
    P1 = np.linalg.inv(S1)
    Pm1 = np.dot(P1, m1)
    P2 = np.linalg.inv(S2)
    Pm2 = np.dot(P2, m2)
    P = P1 + P2
    S = np.linalg.inv(P)
    Pm = Pm1 + Pm2
    m = np.dot(S, Pm)
    return m, S


def test_basic_inference(n_params=2, seed=42):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference
    res = infer.Basic(g, seed=seed)  # need seed for pilot samples
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples
    out = res.run(1000)

    #calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, S_true, atol=0.05)
    assert np.allclose(posterior.xs[0].m, m_true, atol=0.05)


def test_snpe_inference(n_params=2, seed=42):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference
    res = infer.SNPE(g, obs=obs, seed=seed)  # need seed for pilot samples
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples
    out = res.run(n_train=1000, n_rounds=1)

    # calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, S_true, atol=0.05)
    assert np.allclose(posterior.xs[0].m, m_true, atol=0.05)


def test_snpec_inference_mogprop(n_params=2, seed=47):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference
    res = infer.SNPEC(g, obs=obs, seed=seed)  # need seed for pilot samples
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples
    out = res.run(n_train=1000, n_rounds=2, proposal='mog', silent_fail=False)

    # calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, S_true, atol=0.05)
    assert np.allclose(posterior.xs[0].m, m_true, atol=0.05)


def test_snpec_inference_gaussprop(n_params=2, seed=47):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference. need seed for pilot samples
    res = infer.SNPEC(g, obs=obs, seed=seed, n_components=2)
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples. 3 rounds to test re-use of prior and gauss samples
    out = res.run(n_train=1000, n_rounds=3, proposal='gaussian',
                  train_on_all=True, silent_fail=False)

    # calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, S_true, atol=0.05)
    assert np.allclose(posterior.xs[0].m, m_true, atol=0.05)


def test_snpec_inference_discreteprop_mdn(n_params=2, seed=47):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference
    res = infer.SNPEC(g, obs=obs, seed=seed)  # need seed for pilot samples
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples
    out = res.run(n_train=1000, n_rounds=2, proposal='discrete', n_null=10,
                  train_on_all=True, silent_fail=False)

    # calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, S_true, atol=0.05)
    assert np.allclose(posterior.xs[0].m, m_true, atol=0.05)


def test_snpec_inference_discreteprop_mdn_comb(n_params=2, seed=47):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference
    res = infer.SNPEC(g, obs=obs, seed=seed)  # need seed for pilot samples
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples
    out = res.run(n_train=1000, n_rounds=2, proposal='discrete_comb',
                  n_null=10, train_on_all=True, silent_fail=False)

    # calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, S_true, atol=0.05)
    assert np.allclose(posterior.xs[0].m, m_true, atol=0.05)


def test_snpec_inference_discreteprop_maf(n_params=2, seed=47):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference
    rng_maf = np.random
    rng_maf.seed(seed + 50)
    res = infer.SNPEC(g, obs=obs, seed=seed, rng=rng_maf, mode='random',
                      n_mades=5, act_fun='tanh', batch_norm=False)
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples
    out = res.run(n_train=1000, n_rounds=2, proposal='discrete', n_null=10,
                  train_on_all=True, silent_fail=False)

    # calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    posterior.rng = np.random.RandomState(seed=seed + 10)
    posterior_samples = posterior.gen(10000)
    posterior_mean = posterior_samples.mean(axis=0)
    posterior_cov = np.cov(posterior_samples.T)

    assert np.allclose(posterior_cov, S_true, atol=0.05)
    assert np.allclose(posterior_mean, m_true, atol=0.05)


def test_snpec_inference_discreteprop_maf_comb(n_params=2, seed=47):
    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    obs = np.zeros((1, n_params))

    # set up inference
    rng_maf = np.random
    rng_maf.seed(seed + 50)
    res = infer.SNPEC(g, obs=obs, seed=seed, rng=rng_maf, mode='random',
                      n_mades=5, act_fun='tanh', batch_norm=False)
    res.reset(seed=seed)  # reseed generator etc.

    # run with N samples
    out = res.run(n_train=1000, n_rounds=2, proposal='discrete_comb',
                  n_null=10, train_on_all=True, silent_fail=False)

    # calculate true posterior
    m_true, S_true = simplegaussprod(obs, m.noise_cov, p.m, p.S)

    # check result
    posterior = res.predict(obs.reshape(1, -1))
    posterior.rng = np.random.RandomState(seed=seed + 10)
    posterior_samples = posterior.gen(10000)
    posterior_mean = posterior_samples.mean(axis=0)
    posterior_cov = np.cov(posterior_samples.T)
    assert np.allclose(posterior_cov, S_true, atol=0.05)
    assert np.allclose(posterior_mean, m_true, atol=0.05)


def test_snpec_inference_discreteprop_maf_normalize(n_params, seed=47):
    m = Gauss(dim=n_params, noise_cov=0.1)
    p = dd.Uniform(lower=-0.05 * np.ones(n_params),
                   upper=0.05 * np.ones(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)
