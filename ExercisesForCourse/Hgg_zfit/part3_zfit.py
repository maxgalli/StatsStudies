import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
import mplhep as hep
from scipy.integrate import quad
from scipy import interpolate
import zfit
import tensorflow as tf
tf.config.run_functions_eagerly(True)

hep.style.use("CMS")

from utils import plot_as_data, plot_scan
from utils import save_image


print("Systematic uncertainties")

output_dir = "figures_part3_zfit"
nominal_file = "mc_part3_ggH_Tag0.parquet"
file_template = "mc_part3_ggH_Tag0_{}01Sigma.parquet"
var_name = "CMS_hgg_mass"

# get nominal
df_nominal = pd.read_parquet(nominal_file)

# photonID has weights
yield_variations = {}
for sys in ["JEC", "photonID"]:
    for direction in ["Up", "Down"]:
        fl = file_template.format(sys + direction)
        df = pd.read_parquet(fl)
        numerator = df[var_name] * df["weight"]
        denominator = df_nominal[var_name] * df_nominal["weight"]
        yld = numerator.sum() / denominator.sum()
        yld = yld.astype(np.float64)
        print("Systematic varied yield ({}, {}) = {:.3f}".format(sys, direction, yld))
        yield_variations[sys + direction] = yld

# parametric shape uncertainties
mass = zfit.Space(var_name, limits=(100, 180))
higgs_mass = zfit.Parameter("higgs_mass", 125, 124, 126)
sigma = zfit.Parameter("sigma", 2, 1.5, 2.5)
gaus = zfit.pdf.Gauss(mu=higgs_mass, sigma=sigma, obs=mass)

z_mc_nominal = zfit.Data.from_pandas(df_nominal, obs=var_name)
nll = zfit.loss.UnbinnedNLL(gaus, z_mc_nominal)
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result = minimizer.minimize(nll)
result.hesse()
print(result)
best_values = {}
best_values["nominal"] = {p.name: p.numpy() for p in result.params}

for sys in ["scale", "smear"]:
    for direction in ["Up", "Down"]:
        fl = file_template.format(sys + direction)
        df = pd.read_parquet(fl)
        z_mc = zfit.Data.from_pandas(df, obs=var_name)
        nll = zfit.loss.UnbinnedNLL(gaus, z_mc)
        minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
        result = minimizer.minimize(nll)
        result.hesse()
        print(result)
        best_values[sys + direction] = {p.name: p.numpy() for p in result.params}
print(best_values)

# get scale shift
scale_shift_up = (
    np.abs(best_values["scaleUp"]["higgs_mass"] - best_values["nominal"]["higgs_mass"])
    / best_values["nominal"]["higgs_mass"]
)
scale_shift_down = (
    np.abs(
        best_values["scaleDown"]["higgs_mass"] - best_values["nominal"]["higgs_mass"]
    )
    / best_values["nominal"]["higgs_mass"]
)
scale_shift = (scale_shift_up + scale_shift_down) / 2
print("Scale shift = {:.3f}".format(scale_shift))

# get smear shift
smear_shift_up = (
    np.abs(best_values["smearUp"]["sigma"] - best_values["nominal"]["sigma"])
    / best_values["nominal"]["sigma"]
)
smear_shift_down = (
    np.abs(best_values["smearDown"]["sigma"] - best_values["nominal"]["sigma"])
    / best_values["nominal"]["sigma"]
)
smear_shift = (smear_shift_up + smear_shift_down) / 2
print("Smear shift = {:.3f}".format(smear_shift))

# bake into model
d_higgs_mass = zfit.Parameter("dMH", 0, -5, 5)
eta = zfit.Parameter("eta", 0, -5, 5)
#eta.floating = False


def mean_function(higgs_mass, d_higgs_mass, eta):
    return (higgs_mass + d_higgs_mass) * (1 + scale_shift * eta)


mean = zfit.ComposedParameter("mean", mean_function, (higgs_mass, d_higgs_mass, eta))

chi = zfit.Parameter("chi", 0, -5, 5)
#chi.floating = False


def sigma_function(sigma, chi):
    return sigma * (1 + smear_shift * chi)


sigma_full = zfit.ComposedParameter("sigma_full", sigma_function, (sigma, chi))

# start building model

# bkg
fl_data = "data_part1.parquet"
df_data = pd.read_parquet(fl_data)
norm_bkg = zfit.Parameter(
    "model_bkg_Tag0_norm", df_data.shape[0], 0, 3 * df_data.shape[0]
)
lam = zfit.Parameter("lam", -0.05, -0.2, 0)
model_bkg = zfit.pdf.Exponential(lam, obs=mass, extended=norm_bkg)

# signal
xs_ggH = 48.58  # pb
br_hgg = 2.7e-3
df_mc = pd.read_parquet("mc_part1.parquet")
eff = df_mc["weight"].sum() / (xs_ggH * br_hgg)
lumi = 138000
xs_ggH_par = zfit.Parameter("xs_ggH", xs_ggH, 0, 100)
xs_ggH_par.floating = False
br_hgg_par = zfit.Parameter("br_hgg", br_hgg, 0, 1)
br_hgg_par.floating = False
eff_par = zfit.Parameter("eff", eff, 0, 1)
eff_par.floating = False
lumi_par = zfit.Parameter("lumi", lumi, 0, 1e6)
lumi_par.floating = False
r = zfit.Parameter("r", 1, 0, 20)


def yield_multiplicative_factor_function(theta):
    # hot to get the number in the datacard
    return yield_variations["JECUp"]**theta
theta_scale = zfit.Parameter("theta_scale", 0, -5, 5)
yield_multiplicative_factor = zfit.ComposedParameter(
    "yield_multiplicative_factor",
    yield_multiplicative_factor_function,
    (theta_scale,),
)

def yield_multiplicative_factor_asymm_function(theta):
    kappa_up = yield_variations["photonIDUp"]
    kappa_down = yield_variations["photonIDDown"]
    theta_val = theta.value().numpy()
    # see CAT-23-001-paper-v19.pdf pag 7
    if theta_val < -0.5:
        return kappa_down ** (-theta)
    elif theta_val > 0.5:
        return kappa_up**theta
    else:
        return tf.math.exp(
            theta
            * (
                4 * tf.math.log(kappa_up / kappa_down)
                + tf.math.log(kappa_up * kappa_down)
                * (48 * theta**5 - 40 * theta**3 + 15 * theta)
            )
            / 8
        )
theta_smear = zfit.Parameter("theta_smear", 0, -5, 5)
yield_multiplicative_factor_asymm = zfit.ComposedParameter(
    "yield_multiplicative_factor_asymm",
    yield_multiplicative_factor_asymm_function,
    (theta_smear,),
)

def model_ggH_Tag0_norm_function(
    r, xs_ggH, br_hgg, eff, lumi, yield_multiplicative_factor, yield_multiplicative_factor_asymm
    #r, xs_ggH, br_hgg, eff, lumi
    #r, xs_ggH, br_hgg, eff, lumi, yield_multiplicative_factor
):
    return (
        r
        * xs_ggH
        * br_hgg
        * eff
        * lumi
        * yield_multiplicative_factor
        * yield_multiplicative_factor_asymm
    )

model_ggH_Tag0_norm = zfit.ComposedParameter(
    "model_ggH_Tag0_norm",
    model_ggH_Tag0_norm_function,
    (r, xs_ggH_par, br_hgg_par, eff_par, lumi_par, yield_multiplicative_factor, yield_multiplicative_factor_asymm),
    #(r, xs_ggH_par, br_hgg_par, eff_par, lumi_par)
    #(r, xs_ggH_par, br_hgg_par, eff_par, lumi_par, yield_multiplicative_factor),
)
model_ggH_Tag0 = zfit.pdf.Gauss(
    mu=mean, sigma=sigma_full, obs=mass, extended=model_ggH_Tag0_norm
)
model = zfit.pdf.SumPDF([model_bkg, model_ggH_Tag0])
sigma.floating = False
d_higgs_mass.floating = False
# freeze mass
higgs_mass.floating = False

# constraints
# scale, smear, eta, chi
constraint_scale = zfit.constraint.GaussianConstraint(params=theta_scale, observation=0, uncertainty=1)
constraint_smear = zfit.constraint.GaussianConstraint(params=[theta_smear], observation=0, uncertainty=1)
constraint_eta = zfit.constraint.GaussianConstraint(params=[eta], observation=0, uncertainty=1)
constraint_chi = zfit.constraint.GaussianConstraint(params=[chi], observation=0, uncertainty=1)
constraints = [constraint_scale, constraint_smear, constraint_eta, constraint_chi]
#constraints = [constraint_eta, constraint_chi]
#constraints = [constraint_scale, constraint_eta, constraint_chi]

# fit
print("Run fit")
z_data = zfit.Data.from_pandas(df_data, obs=var_name)
nll = zfit.loss.ExtendedUnbinnedNLL(model, z_data, constraints=constraints)
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result = minimizer.minimize(nll)
result.hesse()
print(result)
# get best fit value of nuisance parameters
nuisances_names = ["eta", "chi", "theta_scale", "theta_smear"]
nuisances_bestfit = {}
for p in result.params:
    if p.name in nuisances_names:
        nuisances_bestfit[p.name] = p.numpy()

# scan for r
print("Scanning for r")

denominator = result.loss.value().numpy()
x, y = [], []
for poi in np.linspace(1, 2.5, 10):
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
ax = plot_scan(arr, ax, color="black", var="r", xlims=(0.5, x[-1]), label="With systematics")

# now without systematics
print("Run scan without systematics")

# set values of nuisances to best fit and freeze them
theta_scale.set_value(nuisances_bestfit["theta_scale"])
theta_scale.floating = False
theta_smear.set_value(nuisances_bestfit["theta_smear"])
theta_smear.floating = False
eta.set_value(nuisances_bestfit["eta"])
eta.floating = False
chi.set_value(nuisances_bestfit["chi"])
chi.floating = False

x, y = [], []
for poi in np.linspace(1, 2.5, 10):
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

ax = plot_scan(arr, ax, color="red", var="r", xlims=(0.5, x[-1]), label="Stat-only", y_text_pos=0.85)
for ext in ["png", "pdf"]:
    plt.savefig(f"{output_dir}/part3_scan.{ext}")
