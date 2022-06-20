import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm, expon
from functools import partial


def plot_data(data, nbins=50):
    """
    Plot the data as black points
    """
    hist, edges = np.histogram(data, bins=nbins, density=True)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(bin_centers, hist, 'ko')
    plt.savefig('ulstudy_data.png')


def plot_data_and_model(data, model, res, x_range, nbins=50):
    """
    Plot the data and the model as black points
    """
    #hist, edges = np.histogram(data, bins=nbins, density=True)
    #bin_centers = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(data, bins=nbins, density=True)
    #ax.plot(bin_centers, hist, 'ko')
    ax.plot(x_range, model(x_range, *res.x), 'r-')
    plt.savefig('ulstudy_data_and_model.png')


def plot_clsb(x, y):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, y, 'ko')
    plt.savefig('ulstudy_clsb.png')


def plot_asimov_hist(asimov_hist, asimov_centers, asimov_edges):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(asimov_centers, asimov_hist, width=np.diff(asimov_edges))
    plt.savefig('ulstudy_asimov_hist.png')


def plot_asimov_data(asimov_data, nbins=50):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(asimov_data, bins=nbins, density=True)
    plt.savefig('ulstudy_asimov_data.png')


def plot_asimov_hist_and_model(asimov_hist, asimov_centers, asimov_edges, model, params):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(asimov_centers, asimov_hist, 'ko')
    #print(asimov_hist)
    #print(model(asimov_centers, *params))
    ax.plot(asimov_centers, model(asimov_centers, *params), 'r-')
    #print("aaa")
    #print(quad(lambda x: model(x, *params), 0.1, 3)[0])
    plt.savefig('ulstudy_asimov_hist_and_model.png')


def plot_cl(x, clsb, clb, cls):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, clsb, 'b-')
    ax.plot(x, clb, 'k-')
    ax.plot(x, cls, 'r-')
    plt.savefig('ulstudy_cl.png')


def plot_all(x, clsb, clb, cls, exp_cls):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.fill_between(x, exp_cls[-2], exp_cls[2], color='y')
    ax.fill_between(x, exp_cls[-1], exp_cls[1], color='g')
    ax.plot(x, exp_cls[0], 'k--')
    ax.plot(x, clsb, 'k-')
    ax.plot(x, clsb, 'bo')
    ax.plot(x, clb, 'k-')
    ax.plot(x, clb, 'ko')
    ax.plot(x, cls, 'k-')
    ax.plot(x, cls, 'ro')
    ax.hlines(y=0.05, xmin=x[0], xmax=x[-1], color='r', linestyle='-')
    ax.set_ylabel("p-value")
    ax.set_xlabel("POI")
    plt.savefig('ulstudy_all.png')


class Likelihood:
    def __init__(self, function, data):
        self.function = function
        self.data = data

    def __call__(self, params):
        return np.prod(self.function(self.data, *params))


class NLL(Likelihood):
    def __call__(self, params):
        return -np.sum([np.log(self.function(self.data, *params))])


class BinnedLikelihood(Likelihood):
    def __init__(self, function, hist, edges):
        self.function = function
        self.hist = hist
        self.edges = edges

    def __call__(self, params):
        integration_limits = []
        inf_limit = self.edges[0]
        for sup_limit in self.edges[1:]:
            integration_limits.append((inf_limit, sup_limit))
            inf_limit = sup_limit
        n_tot = self.hist.sum()
        nu_is = np.array([n_tot * quad(lambda x: self.function(x, *params), inf_lim, sup_lim)[0] for inf_lim, sup_lim in integration_limits])
        return - np.sum(self.hist * np.log(nu_is))


# TODO: extended
#def model(x, f_sig, mu, sigma, tau):
def model(x, f_sig, tau):
    mu = 1.2
    sigma = 0.1
    return f_sig*(1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))) + (1-f_sig)*((1/tau) * np.exp(-(x / tau)))
    #return f_sig * norm(mu, sigma).pdf(x) + (1 - f_sig) * expon(scale=tau).pdf(x)
    #return f_sig * norm_numba.pdf(x, mu, sigma) + (1 - f_sig) * expon_numba.pdf(x, tau)


def n_evs(f, n_tot):
    return f * n_tot


def q_mu(nll1, nll2, poi1, poi2):
    q = 2 * (nll1 - nll2)
    zeros = np.zeros(q.shape)
    condition = (poi2 > poi1) | (q < 0)
    return np.where(condition, zeros, q)


def p_mu(q_mu, nsigma=0):
    return 1 - norm.cdf(np.sqrt(q_mu) - nsigma)


def p_alt(q_obs, q_alt):
    sqrtqobs = np.sqrt(q_obs)
    sqrtqalt = np.sqrt(q_alt)
    return 1.0 - norm.cdf(sqrtqobs - sqrtqalt)


def generate_asimov_hist(model, params, bounds, nbins=10000):
#def generate_asimov_hist(model, params, bounds, nbins=100):
    bin_edges = np.linspace(bounds[0], bounds[1], nbins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    integration_limits = []
    inf_limit = bin_edges[0]
    for sup_limit in bin_edges[1:]:
        integration_limits.append((inf_limit, sup_limit))
        inf_limit = sup_limit
    hist = np.array([quad(lambda x: model(x, *params), inf_lim, sup_lim)[0] for inf_lim, sup_lim in integration_limits])

    return hist, bin_edges, bin_centers


def generate_pseudo_asimov_dataset(model, params, bounds, nevs=1000000, nbins=100):
    hist, bin_edges, bin_centers = generate_asimov_hist(model, params, bounds, nbins)
    edges_pairs = list(zip(bin_edges[:-1], bin_edges[1:]))
    return np.concatenate([np.random.uniform(ep[0], ep[1], int(np.round(nevs*h))) for ep, h in zip(edges_pairs, hist)])


def dataset_from_hist(hist, bin_edges, nevs=10000):
    # un po' una scopata mentale
    bin_centers = bin_edges[0:-1] + np.diff(bin_edges) / 2
    arr = np.concatenate([center * np.ones(int(h)) for center, h in zip(bin_centers, np.rint(nevs*0.01*hist))])
    np.random.shuffle(arr)
    return arr


def main():
    bounds = (0.1, 3.0)
    bnds  = ((0.001, 1.0), (0.03, 1.0)) #to feed the minimizer

    bkg = np.random.exponential(0.5, 300)
    peak = np.random.normal(1.2, 0.1, 10)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    #np.save('data.npy', data)
    #h, e = np.histogram(data, bins=100, range=bounds)

    plot_data(data)

    lk = NLL(model, data)
    #lk = BinnedLikelihood(model, h, e)
    #x0 = [300/(300+10), 1.2, 0.1, 0.5]
    x0 = [10/(300+10), 0.5]
    global_res = minimize(fun=lk, x0=x0, method='Powell', bounds=bnds, tol=1e-6)
    print(f"Best fit to data: {global_res.x}")
    plot_data_and_model(data, model, global_res, np.linspace(bounds[0], bounds[1], 1000))

    pois_null = np.linspace(0.001, 0.2, 30)
    pois_best = np.ones(pois_null.shape) * global_res.x[0]

    nll_best = np.ones(pois_best.shape) * global_res.fun
    nll_null = []
    for pn in pois_null:
        def to_minimize(params):
            return lk([pn, *params])
        x0 = global_res.x[1:] 
        res = minimize(fun=to_minimize, x0=x0, method='Nelder-Mead', bounds=bnds, tol=1e-6)
        nll_null.append(res.fun)

    qobs = q_mu(nll_null, nll_best, pois_null, pois_best)
    pobs = p_mu(qobs)
    print("nll_null: ", nll_null)
    print("nll_best: ", nll_best)
    print("qobs: ", qobs)
    print("pobs: ", pobs)

    plot_clsb(pois_null, pobs)

    # Asimov
    print("Asimov stuff")
    bkg_only_pars = [0.0] + list(global_res.x[1:])
    asimov_hist, asimov_edges, asimov_centers = generate_asimov_hist(model, bkg_only_pars, bounds)
    asimov_dataset = generate_pseudo_asimov_dataset(model, bkg_only_pars, bounds, nevs=310)
    plot_asimov_hist(asimov_hist, asimov_centers, asimov_edges)
    #asimov_data = dataset_from_hist(asimov_hist, asimov_edges)
    plot_asimov_data(asimov_dataset)

    #lk_asimov = BinnedLikelihood(model, asimov_hist, asimov_edges)
    lk_asimov = NLL(model, asimov_dataset)
    #lk_asimov = NLL(model, asimov_data)
    def to_minimize(params):
        return lk_asimov([0.0, *params])
    x0 = global_res.x[1:]
    global_res_asimov = minimize(fun=to_minimize, x0=x0, method='Nelder-Mead', bounds=bnds[1:], tol=1e-6)
    #global_res_asimov = minimize(fun=lk_asimov, x0=global_res.x, method='Nelder-Mead', tol=1e-6)
    print(f"Best nuisance values in Asimov {global_res_asimov.x}")
    #plot_asimov_hist_and_model(asimov_hist, asimov_centers, asimov_edges, model, global_res_asimov.x)

    pois_alt = np.zeros(pois_null.shape)
    nll_best_asimov = np.ones(pois_best.shape) * global_res_asimov.fun
    nll_null_asimov = []
    for pn in pois_null:
        def to_minimize_loc(params):
            return lk_asimov([pn, *params])
        #x0_loc = global_res_asimov.x[1:] 
        x0_loc = global_res_asimov.x
        res = minimize(fun=to_minimize_loc, x0=x0_loc, method='Nelder-Mead', bounds=bnds[1:], tol=1e-6)
        nll_null_asimov.append(res.fun)
        print(f"poi {pn} best {res.x}")
    q_asimov = q_mu(nll_null_asimov, nll_best_asimov, pois_null, pois_alt)
    p_asimov = p_alt(qobs, q_asimov)
    print("nll_null_asimov: ", nll_null_asimov)
    print("nll_best_asimov: ", nll_best_asimov)
    print("q_asimov: ", q_asimov)
    print("p_asimov: ", p_asimov)
    cls = pobs / p_asimov
    plot_cl(pois_null, pobs, p_asimov, cls)

    exp_clsb = {}
    exp_clb = {}
    exp_cls = {}
    sigmas = [0, 1, 2, -1, -2]
    for sigma in sigmas:
        exp_clsb[sigma] = p_mu(q_asimov, sigma)
        exp_clb[sigma] = np.ones(exp_clsb[sigma].shape) * norm.cdf(sigma)
        exp_cls[sigma] = exp_clsb[sigma] / exp_clb[sigma]
    print("exp_clsb: ", exp_clsb)
    print("exp_clb: ", exp_clb)
    print("exp_cls: ", exp_cls)
    plot_all(pois_null, pobs, p_asimov, cls, exp_cls)


if __name__ == '__main__':
    main()