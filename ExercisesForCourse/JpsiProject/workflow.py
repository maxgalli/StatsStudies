from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import zfit
from hepstats.hypotests import UpperLimit, Discovery
from hepstats.hypotests.parameters import POI
from hepstats.hypotests.parameters import POIarray
from hepstats.hypotests.calculators import AsymptoticCalculator
from hepstats.hypotests.calculators import FrequentistCalculator
from utils_zfit import plotlimit
from scipy import interpolate
import mplhep as hep

hep.style.use("CMS")


def plot_scan(arr, ax, color, label=None):
    ax.plot(arr[0], arr[1], color=color, label=label)
    minimum = arr[:, np.argmin(arr[1])]
    level = 1.0
    level = minimum[1] + level
    level_arr = np.ones(len(arr[1])) * level
    # Get index of the two points in poi_values where the NLL crosses the horizontal line at 1
    indices = np.argwhere(
        np.diff(np.sign(arr[1] - level_arr))
    ).flatten()
    points = [arr[:, i] for i in indices]
    for point in points:
        ax.plot([point[0], point[0]], [minimum[1], point[1]], color="k", linestyle="--")
 
    ax.set_xlabel("$xs \psi(2S)$")
    ax.set_ylabel("-2$\Delta$lnL")
    ax.set_ylim(0, 10)
    ax.set_xlim(5, 18)
    ax.axhline(1.0, color="k", linestyle="--")
   
    return ax


def main():
    # Exercise 0
    print("Exercise 0: fit lower statistics sample")
    data = pkl.load(open("DataSet_lowstat.pkl", "rb"))
    mass = zfit.Space("mass", (2, 6))
    zdata = zfit.Data.from_numpy(obs=mass, array=data)

    mean_jpsi = zfit.Parameter("mean_jpsi", 3.1, 2.8, 3.4)
    sigma_jpsi = zfit.Parameter("sigma_jpsi", 0.3, 0.0001, 1.)
    alpha_jpsi = zfit.Parameter("alpha_jpsi", 1.5, -5., 5.)
    n_jpsi = zfit.Parameter("n_jpsi", 1.5, 0.5, 10.)
    pdf_jpsi = zfit.pdf.CrystalBall(mu=mean_jpsi, sigma=sigma_jpsi, alpha=alpha_jpsi, n=n_jpsi, obs=mass)

    mean_psi2s = zfit.Parameter("mean_psi2s", 3.7, 2.5, 3.95)
    pdf_psi2s = zfit.pdf.CrystalBall(mu=mean_psi2s, sigma=sigma_jpsi, alpha=alpha_jpsi, n=n_jpsi, obs=mass)

    a1 = zfit.Parameter("a1", -0.7, -2, 2)
    a2 = zfit.Parameter("a2", 0.3, -2, 2)
    a3 = zfit.Parameter("a3", -0.03, -2, 2)
    pdf_bkg = zfit.pdf.Chebyshev2(obs=mass, coeffs=[a1, a2, a3])

    yield_jpsi = zfit.Parameter("yield_jpsi", 1500, 0, 10000)
    eff_psi2s = zfit.Parameter("eff_psi2s", 0.75, 0.00001, 1., floating=False)
    lumi_psi2s = zfit.Parameter("lumi_psi2s", 0.64, 0.00001, 50., floating=False)
    cross_psi2s = zfit.Parameter("cross_psi2s", 3., 0., 40.)
    yield_psi2s = zfit.ComposedParameter("yield_psi2s", lambda x, y, z: x*y*z, (eff_psi2s, lumi_psi2s, cross_psi2s))
    yield_bkg = zfit.Parameter("yield_bkg", 5000, 0, 50000)

    extended_jpsi = pdf_jpsi.create_extended(yield_jpsi)
    extended_psi2s = pdf_psi2s.create_extended(yield_psi2s)
    extended_bkg = pdf_bkg.create_extended(yield_bkg)

    model = zfit.pdf.SumPDF([extended_jpsi, extended_psi2s, extended_bkg])

    nll = zfit.loss.ExtendedUnbinnedNLL(model, zdata)
    minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
    result = minimizer.minimize(nll)
    result.hesse()
    print(result.params)

    n_bins = 100
    bins, edges = np.histogram(data, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots()
    ax.errorbar(centers, bins, yerr=np.sqrt(bins), fmt='o', label="Data", color="black")
    x = np.linspace(2, 6, 1000)
    ax.set_xlabel("$m_{\mu\mu}$ [GeV]")
    ax.set_ylabel("Events / [{} GeV]".format((6 - 2) / n_bins))
    ax.set_ylim(0, 80)
    ax.legend()
    plt.savefig("images/Exercise0_data.png")
    ax.plot(x, data.shape[0] / n_bins * (6 - 2) * model.pdf(x), label="Model", color="red")
    ax.legend()
    plt.savefig("images/Exercise0.png")

    # Exercise 2
    print("Exercise 2: significance")
    calculator = AsymptoticCalculator(input=nll, minimizer=minimizer)
    sig_yield_poi = POI(cross_psi2s, 0)
    discovery = Discovery(calculator=calculator, poinull=sig_yield_poi)
    discovery.result()

    # Exercise 3
    print("Exercise 3: upper limit")
    calculator.bestfit = result

    # the null hypothesis
    bkg_only = POI(cross_psi2s, 0)
    sig_yield_scan = POIarray(cross_psi2s, np.linspace(0, 40, 20))

    ul = UpperLimit(calculator=calculator, poinull=sig_yield_scan, poialt=bkg_only)
    ul.upperlimit(alpha=0.05)

    fig, ax = plt.subplots()
    ax = plotlimit(ul, CLs=False)
    ax.set_xlabel("$xs \psi(2S)$")
    plt.savefig("images/Exercise3.png")

    # Exercise 4
    print("Exercise 4: higher statistics sample")
    data = pkl.load(open("DataSet.pkl", "rb"))
    zdata = zfit.Data.from_numpy(obs=mass, array=data)

    lumi_psi2s.set_value(37)
    lumi_psi2s.floating = False
    nll = zfit.loss.ExtendedUnbinnedNLL(model, zdata)
    result = minimizer.minimize(nll)

    denominator = result.loss.value().numpy()
    x = []
    y = []
    for poi in np.linspace(0, 40, 10):
        cross_psi2s.set_value(poi)
        cross_psi2s.floating = False
        result_scan = minimizer.minimize(nll)
        numerator = result_scan.loss.value().numpy()
        twonll = 2 * (numerator - denominator)
        x.append(poi)
        y.append(twonll)
    x = np.array(x)
    y = np.array(y)
    print(x)
    print(y)

    func = interpolate.interp1d(x, y, kind='cubic')
    n_interp = 1000
    x_interp = np.linspace(0, 40, n_interp)
    y_interp = func(x_interp)
    y_interp = y_interp - np.min(y_interp)
    arr = np.array([x_interp, y_interp])
    #arr = arr[:, arr[1] < 10]
    
    fig, ax = plt.subplots()
    ax = plot_scan(arr, ax, color="black")
    plt.savefig("images/Exercise4.png")

    # Exercise 5
    print("Exercise 5: add systematic uncertainty")
    sig_eff = zfit.Parameter("sig_eff", 1., 0., 3.)
    constraint = zfit.constraint.GaussianConstraint(sig_eff, observation=1., uncertainty=0.1)
    yield_psi2s_mod = zfit.ComposedParameter("yield_psi2s_mod", lambda x, y, z, k: x*y*z*k, (eff_psi2s, lumi_psi2s, cross_psi2s, sig_eff))
    extended_psi2s_mod = pdf_psi2s.create_extended(yield_psi2s_mod)
    model_mod = zfit.pdf.SumPDF([extended_jpsi, extended_psi2s_mod, extended_bkg])

    nll = zfit.loss.ExtendedUnbinnedNLL(model_mod, zdata, constraints=constraint)
    minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
    result = minimizer.minimize(nll)

    result = minimizer.minimize(nll)

    denominator = result.loss.value().numpy()
    x_mod = []
    y_mod = []
    for poi in np.linspace(0, 40, 10):
        cross_psi2s.set_value(poi)
        cross_psi2s.floating = False
        result_scan = minimizer.minimize(nll)
        numerator = result_scan.loss.value().numpy()
        twonll = 2 * (numerator - denominator)
        x_mod.append(poi)
        y_mod.append(twonll)
    x_mod = np.array(x_mod)
    y_mod = np.array(y_mod)
    print(x_mod)
    print(y_mod)

    func_mod = interpolate.interp1d(x_mod, y_mod, kind='cubic')
    x_mod_interp = np.linspace(0, 40, n_interp)
    y_mod_interp = func_mod(x_mod_interp)
    y_mod_interp = y_mod_interp - np.min(y_mod_interp)
    arr_mod = np.array([x_mod_interp, y_mod_interp])
    #arr_mod = arr_mod[:, arr_mod[1] < 10]
    
    fig, ax = plt.subplots()
    ax = plot_scan(arr, ax, color="black", label="Without systematic uncertainty")
    ax = plot_scan(arr_mod, ax, color="red", label="With systematic uncertainty")
    ax.legend()
    plt.savefig("images/Exercise5.png")


if __name__ == '__main__':
    main()