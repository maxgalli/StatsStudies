import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import interpolate


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--lk",
        required=True,
        type=str,
        choices=["unbinned", "binned"],
        help="Type of likelihood",
    )

    parser.add_argument(
        "--hypo",
        required=True,
        type=str,
        choices=["asymptotic", "toys"],
        help="Type of hypothesis test",
    )

    return parser.parse_args()


def plot_data_and_model(data, model, res, x_range, nbins=50):
    """
    Plot the data and the model
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(data, bins=nbins, density=True)
    ax.plot(x_range, model(x_range, *res.x), 'r-')
    plt.savefig('ulstudy_data_and_model.png')


def plot_asimov_hist_and_model(asimov_hist, asimov_centers, asimov_edges, model, params):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(asimov_centers, asimov_hist, 'ko')
    ax.plot(asimov_centers, model(asimov_centers, *params), 'r-')
    plt.savefig('ulstudy_asimov_hist_and_model.png')


def plot_cl(x, clsb, clb, cls):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, clsb, 'b-')
    ax.plot(x, clb, 'k-')
    ax.plot(x, cls, 'r-')
    plt.savefig('ulstudy_cl.png')


def plot_all(x, clsb, clb, cls, exp_cls, lk, hypo):
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
    plt.savefig(f'ulstudy_all_{lk}_{hypo}.png')


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
def model(x, f_sig, tau):
    mu = 1.2
    sigma = 0.1
    return f_sig*(1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))) + (1-f_sig)*((1/tau) * np.exp(-(x / tau)))
    #return f_sig * norm(mu, sigma).pdf(x) + (1 - f_sig) * expon(scale=tau).pdf(x)


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


def generate_asimov_hist(model, params, bounds, nbins=100):
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


def main(args):
    bounds = (0.1, 3.0)
    bnds  = ((0.001, 1.0), (0.03, 1.0)) #to feed the minimizer

    # Generate dataset with very small signal
    bkg = np.random.exponential(0.5, 300)
    peak = np.random.normal(1.2, 0.1, 10)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    if args.lk == "binned":
        h, e = np.histogram(data, bins=100, range=bounds)

    if args.lk == "binned":
        lk = BinnedLikelihood(model, h, e)
    else:
        lk = NLL(model, data)
    x0 = [10/(300+10), 0.5]
    global_res = minimize(fun=lk, x0=x0, method='Powell', bounds=bnds, tol=1e-6)
    plot_data_and_model(data, model, global_res, np.linspace(bounds[0], bounds[1], 1000))

    pois_null = np.linspace(0.001, 0.2, 30)
    pois_best = np.ones(pois_null.shape) * global_res.x[0]

    nll_best = np.ones(pois_best.shape) * global_res.fun
    nll_null = []
    for pn in pois_null:
        def to_minimize(params):
            return lk([pn, *params])
        x0 = global_res.x[1:] 
        res = minimize(fun=to_minimize, x0=x0, method='Powell', bounds=bnds, tol=1e-6)
        nll_null.append(res.fun)

    qobs = q_mu(nll_null, nll_best, pois_null, pois_best)
    pobs = p_mu(qobs)

    # Asimov
    bkg_only_pars = [0.0] + list(global_res.x[1:])
    if args.lk == "binned":
        asimov_hist, asimov_edges, asimov_centers = generate_asimov_hist(model, bkg_only_pars, bounds, nbins=100)
        asimov_hist *= 310
    else:
        asimov_dataset = generate_pseudo_asimov_dataset(model, bkg_only_pars, bounds, nevs=310)

    if args.lk == "binned":
        lk_asimov = BinnedLikelihood(model, asimov_hist, asimov_edges)
    else:
        lk_asimov = NLL(model, asimov_dataset)
    def to_minimize(params):
        return lk_asimov([0.0, *params])
    x0 = global_res.x[1:]
    global_res_asimov = minimize(fun=to_minimize, x0=x0, method='Powell', bounds=bnds[1:], tol=1e-6)

    pois_alt = np.zeros(pois_null.shape)
    nll_best_asimov = np.ones(pois_best.shape) * global_res_asimov.fun
    nll_null_asimov = []
    for pn in pois_null:
        def to_minimize_loc(params):
            return lk_asimov([pn, *params])
        x0_loc = global_res_asimov.x
        res = minimize(fun=to_minimize_loc, x0=x0_loc, method='Powell', bounds=bnds[1:], tol=1e-6)
        nll_null_asimov.append(res.fun)
    q_asimov = q_mu(nll_null_asimov, nll_best_asimov, pois_null, pois_alt)
    p_asimov = p_alt(qobs, q_asimov)
    cls = pobs / p_asimov
    plot_cl(pois_null, pobs, p_asimov, cls)

    # Expected
    exp_clsb = {}
    exp_clb = {}
    exp_cls = {}
    sigmas = [0, 1, 2, -1, -2]
    for sigma in sigmas:
        exp_clsb[sigma] = p_mu(q_asimov, sigma)
        exp_clb[sigma] = np.ones(exp_clsb[sigma].shape) * norm.cdf(sigma)
        exp_cls[sigma] = exp_clsb[sigma] / exp_clb[sigma]
    plot_all(pois_null, pobs, p_asimov, cls, exp_cls, args.lk, args.hypo)

    # Find upper limit
    interpolated_funcs = {}
    upper_limits = {}
    interpolated_funcs["obs"] = interpolate.interp1d(pois_null, cls, kind="cubic")
    for sigma in sigmas:
        interpolated_funcs["exp_{}".format(str(sigma).replace("-", "m"))] = interpolate.interp1d(pois_null, exp_cls[sigma], kind="cubic")
    more_pois_null = np.linspace(0.001, 0.2, 10000)
    line = np.ones(more_pois_null.shape) * 0.05
    for name, func in interpolated_funcs.items():
        interpolated_line = func(more_pois_null)
        idx = np.argwhere(np.diff(np.sign(interpolated_line - line))).flatten()
        upper_limits[name] = more_pois_null[idx]
    print("upper_limits: ", upper_limits)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)