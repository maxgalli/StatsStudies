import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
import mplhep as hep
from scipy.integrate import quad
from scipy import interpolate
import zfit

hep.style.use("CMS")

from utils import plot_as_data, plot_scan
from utils import save_image


# Signal modelling 
print("Checking MC")
fl = "mc_part1.parquet"
var_name = "CMS_hgg_mass"
output_dir = "figures_part1_zfit"
df = pd.read_parquet(fl)

fig, ax = plt.subplots()
ax = plot_as_data(df[var_name], nbins=100, ax=ax)
ax.set_xlabel("$m_{\gamma\gamma}$ [GeV]")
save_image("part1_signal_mass", output_dir)

# Fit Gaussian to MC events and plot
print("Fitting Gaussian to MC")

mass = zfit.Space(var_name, limits=(100, 180))
zdata = zfit.Data.from_pandas(df, obs=var_name)

higgs_mass = zfit.Parameter("higgs_mass", 125, 120, 130)
higgs_mass.floating = False
sigma = zfit.Parameter("sigma", 2, 1, 5)
model = zfit.pdf.Gauss(mu=higgs_mass, sigma=sigma, obs=mass)

nll = zfit.loss.UnbinnedNLL(model, zdata)
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result = minimizer.minimize(nll)
result.hesse()
print(result)

fig, ax = plt.subplots()
ax = plot_as_data(df[var_name], ax=ax)
mn, mx = df[var_name].min(), df[var_name].max()
x = np.linspace(mn, mx, 1000)
y = model.pdf(x, norm_range=mass)
ax.plot(x, y, label="fit")
ax.set_xlabel("$m_{\gamma\gamma}$ [GeV]")
ax.legend()
save_image("part1_signal_model_v0", output_dir)

# define a model with a variable mean
print("Fitting Gaussian to MC with variable mean")

d_higgs_mass = zfit.Parameter("dMH", 0, -1, 1)
def mean_function(higgs_mass, d_higgs_mass):
    return higgs_mass + d_higgs_mass
mean = zfit.ComposedParameter("mean", mean_function, (higgs_mass, d_higgs_mass))
model = zfit.pdf.Gauss(mu=mean, sigma=sigma, obs=mass)

nll = zfit.loss.UnbinnedNLL(model, zdata)
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result = minimizer.minimize(nll)
result.hesse()
print(result)

fig, ax = plt.subplots()
ax = plot_as_data(df[var_name], ax=ax)
mn, mx = df[var_name].min(), df[var_name].max()
x = np.linspace(mn, mx, 1000)
y = model.pdf(x, norm_range=mass)
ax.plot(x, y, label="fit")
ax.set_xlabel("$m_{\gamma\gamma}$ [GeV]")
ax.legend()
save_image("part1_signal_model_v1", output_dir)

# Signal normalisation
print("Getting signal normalisation")
xs_ggH = 48.58  # pb
br_hgg = 2.7e-3

# this part is slightly different from what jon does
sumw = df["weight"].sum()
eff = sumw / (xs_ggH * br_hgg)
print(f"Efficiency of ggH events landing in Tag0 is: {eff:.5f}")

lumi = 138000
n = eff * xs_ggH * br_hgg * lumi
print(f"For 138 fb^-1, the expected number of ggH events is: N = xs * BR * eff * lumi = {n:.5f}")

# Bkg modelling
print("Background modelling from sidebands in data")
fl_data = "data_part1.parquet"
df_data = pd.read_parquet(fl_data)
df_data_sides = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 115) | (df_data[var_name] > 135) & (df_data[var_name] < 180)]

mass_left = zfit.Space("CMS_hgg_mass", limits=(100, 115))
mass_right = zfit.Space("CMS_hgg_mass", limits=(135, 180))
mass_full = mass_left + mass_right
zdata_data = zfit.Data.from_pandas(df_data, obs=var_name)
lam = zfit.Parameter("lam", -0.05, -0.2, 0)
model_bkg = zfit.pdf.Exponential(lam, obs=mass_full)

nll = zfit.loss.UnbinnedNLL(model_bkg, zdata_data)
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result = minimizer.minimize(nll)
result.hesse()
print(result)

fig, ax = plt.subplots()
ax = plot_as_data(df_data_sides[var_name], nbins=80, normalize_to=len(df_data), ax=ax)
ax.set_xlabel("$m_{\gamma\gamma}$ [GeV]")
x = np.linspace(100, 180, 1000)
y = model_bkg.pdf(x, norm_range=mass)
ax.plot(x, y, label="fit")
ax.legend()
save_image("part1_data_sidebands", output_dir)

# extension: signal model normalization
print("Extension: signal model normalization")
xs_ggH_par = zfit.Parameter("xs_ggH", xs_ggH, 0, 100)
xs_ggH_par.floating = False
br_hgg_par = zfit.Parameter("br_hgg", br_hgg, 0, 1)
br_hgg_par.floating = False
eff_par = zfit.Parameter("eff", eff, 0, 1)
eff_par.floating = False
lumi_par = zfit.Parameter("lumi", lumi, 0, 1e6)
lumi_par.floating = False

################
# starting part 2
################

# Datacard simulation
print("Datacard simulation")

# bkg
# define bkg model normalization term
norm_bkg = zfit.Parameter("model_bkg_Tag0_norm", df_data.shape[0], 0, 3 * df_data.shape[0])
model_bkg_card = zfit.pdf.Exponential(lam, obs=mass, extended=norm_bkg)

# sig 
r = zfit.Parameter("r", 1, 0, 20)
def model_ggH_Tag0_norm_function(r, xs_ggH, br_hgg, eff, lumi):
    return r * xs_ggH * br_hgg * eff * lumi
model_ggH_Tag0_norm = zfit.ComposedParameter("model_ggH_Tag0_norm", model_ggH_Tag0_norm_function, (r, xs_ggH_par, br_hgg_par, eff_par, lumi_par))
model_ggH_card = zfit.pdf.Gauss(mu=mean, sigma=sigma, obs=mass, extended=model_ggH_Tag0_norm)

# total
model_card = zfit.pdf.SumPDF([model_bkg_card, model_ggH_card])

print("Fitting datacard model to data")
sigma.floating = False
d_higgs_mass.floating = False
nll = zfit.loss.ExtendedUnbinnedNLL(model_card, zdata_data)
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result = minimizer.minimize(nll)
result.hesse()
print(result)
# get best fit value for r
best_r = r.value().numpy()

# scan for r
print("Scanning for r")

denominator = result.loss.value().numpy()
x, y = [], []
for poi in np.linspace(1, 2.5, 20):
    r.set_value(poi)
    r.floating = False
    result_scan = minimizer.minimize(nll)
    numerator = result_scan.loss.value().numpy()
    twonll = 2 * (numerator - denominator)
    x.append(poi)
    y.append(twonll)
x = np.array(x)
y = np.array(y)
print(x)
print(y)

func = interpolate.interp1d(x, y, kind="cubic")
n_interp = 1000
x_interp = np.linspace(x[0], x[-1], n_interp)
y_interp = func(x_interp)
y_interp = y_interp - np.min(y_interp)
arr = np.array([x_interp, y_interp])

fig, ax = plt.subplots()
ax = plot_scan(arr, ax, color="black", var="r", xlims=(0.5, x[-1]), label="r")
for ext in ["png", "pdf"]:
    plt.savefig(f"{output_dir}/part2_scan.{ext}")

# plot prefit and postfit superimposed to data
print("Plotting prefit and postfit superimposed to data")
fig, ax = plt.subplots()
ax = plot_as_data(df_data[var_name], nbins=80, ax=ax)
ax.set_xlabel("$m_{\gamma\gamma}$ [GeV]")
x = np.linspace(100, 180, 1000)
r.set_value(best_r)
y = model_card.pdf(x, norm_range=mass)
ax.plot(x, y, label=f"Postfit S + B model (r = {best_r:.3f})", color="r")
# now set r = 1 and plot
r.set_value(1)
y = model_card.pdf(x, norm_range=mass)
ax.plot(x, y, label="Prefit S + B model (r = 1)", color="b")
ax.legend()
save_image("part2_sb_model", output_dir)