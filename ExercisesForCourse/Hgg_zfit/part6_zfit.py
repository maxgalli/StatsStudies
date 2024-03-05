import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
import mplhep as hep
from scipy.integrate import quad
from scipy import interpolate
from scipy.interpolate import griddata
import zfit

hep.style.use("CMS")

from utils import plot_as_data, plot_scan, plot_as_heatmap, plot_as_contour
from utils import save_image

output_dir = "figures_part6_zfit"

procs = ['ggH','VBF']
cats = ['Tag0','Tag1']

mc_template = "mc_part6_{}.parquet"

mc, mc_df, dmh, mean, sigma, model = {}, {}, {}, {}, {}, {}
xs, eff, norm, r, model_sig_norm = {}, {}, {}, {}, {}

var_name = "CMS_hgg_mass"
mass = zfit.Space(var_name, limits=(100, 180))
higgs_mass = zfit.Parameter("higgs_mass", 125, 120, 130)
higgs_mass.floating = False

def mean_function(higgs_mass, d_higgs_mass):
    return higgs_mass + d_higgs_mass

def norm_function(r, xs, br, eff, lumi):
    return r * xs * br * eff * lumi

# signal models
print("Signal modelling")
for cat in cats:
    for proc in procs:
        key = f"{proc}_{cat}"
        print(f"Fit signal model for {key}")
        fl = mc_template.format(key)
        mc_df[key] = pd.read_parquet(fl)
        mc[key] = zfit.Data.from_pandas(mc_df[key], obs=var_name) 
        dmh[key] = zfit.Parameter(f"dMH_{key}", 0, -1, 1)
        mean[key] = zfit.ComposedParameter(f"mean_{key}", mean_function, (higgs_mass, dmh[key]))
        sigma[key] = zfit.Parameter(f"sigma_{key}", 2, 1, 5)
        model[key] = zfit.pdf.Gauss(mu=mean[key], sigma=sigma[key], obs=mass)

        nll = zfit.loss.UnbinnedNLL(model[key], mc[key])
        minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
        result = minimizer.minimize(nll)
        result.hesse()
        print(result)

xs_ggH = 48.58
xs_VBF = 3.782
br_hgg = 2.7e-3
lumi = 138000

xs["ggH"] = zfit.Parameter("xs_ggH", xs_ggH, 0, 100)
xs["ggH"].floating = False
xs["VBF"] = zfit.Parameter("xs_VBF", xs_VBF, 0, 100)
xs["VBF"].floating = False
br_gamgam = zfit.Parameter("BR_gamgam", br_hgg, 0, 1)
br_gamgam.floating = False
lumi = zfit.Parameter("lumi", lumi, 0, 1e6)
lumi.floating = False
for proc in procs:
    r[proc] = zfit.Parameter(f"r_{proc}", 1, -2, 20)
    for cat in cats:
        key = f"{proc}_{cat}"
        sumw = mc_df[key]["weight"].sum()
        eff_val = sumw / (xs[proc] * br_gamgam)
        eff[key] = zfit.Parameter(f"eff_{key}", eff_val, 0, 1)
        eff[key].floating = False
        norm[key] = zfit.ComposedParameter(f"norm_{key}", norm_function, (r[proc], xs[proc], br_gamgam, eff[key], lumi))

for cat in cats:
    model_sig_norm[cat] = {}
    for proc in procs:
        key = f"{proc}_{cat}"
        #model_sig_norm[cat][proc] = model[key].create_extended(norm[key])
        model_sig_norm[cat][proc] = zfit.pdf.Gauss(mu=mean[key], sigma=sigma[key], obs=mass, extended=norm[key])

# background models
print("Background modelling")
data_template = "data_part6_data_{}.parquet"
data, alpha, norm_bkg, model_bkg = {}, {}, {}, {}

mass_left = zfit.Space(var_name, limits=(110, 115))
mass_right = zfit.Space(var_name, limits=(135, 180))
mass_full = mass_left + mass_right
for cat in cats:
    print(f"Fit background model for {key}")
    fl = data_template.format(cat)
    df = pd.read_parquet(fl)
    data[cat] = zfit.Data.from_pandas(df, obs=var_name)
    alpha[cat] = zfit.Parameter(f"alpha_{cat}", -0.05, -0.2, 0)
    mod = zfit.pdf.Exponential(alpha[cat], obs=mass_full)
    nll = zfit.loss.UnbinnedNLL(mod, data[cat])
    minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
    result = minimizer.minimize(nll)
    result.hesse()
    print(result)

    norm_bkg[cat] = zfit.Parameter(f"norm_bkg_{cat}", df.shape[0], 0, 3 * df.shape[0])
    model_bkg[cat] = zfit.pdf.Exponential(alpha[cat], obs=mass, extended=norm_bkg[cat])

# combined model
print("Combine models")
model_combined = {}
nlls = {}
def simultaneous_nll(*nlls):
    return sum(nlls)
for cat in cats:
    model_combined[cat] = zfit.pdf.SumPDF([model_sig_norm[cat][proc] for proc in procs] + [model_bkg[cat]])
    nlls[cat] = zfit.loss.ExtendedUnbinnedNLL(model_combined[cat], data[cat])
print(nlls)
#nll = simultaneous_nll(*nlls.values())
nll = nlls["Tag0"] + nlls["Tag1"]
for s in sigma.values():
    s.floating = False
for d in dmh.values():
    d.floating = False
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result = minimizer.minimize(nll)
result.hesse()
print(result)

# scan 2D
# r_ggH in 0.5, 2.5
# r_VBF in -1, 2
minimum = (result.params["r_ggH"]["value"], result.params["r_VBF"]["value"])
denominator = result.loss.value().numpy()
x, y, z = [], [], []

for r_ggH in np.linspace(0.5, 2.5, 30):
    for r_VBF in np.linspace(-1, 2, 30):
        r["ggH"].set_value(r_ggH)
        r["VBF"].set_value(r_VBF)
        r["ggH"].floating = False
        r["VBF"].floating = False
        numerator = result.loss.value().numpy()
        twonll = 2 * (numerator - denominator)
        x.append(r_ggH)
        y.append(r_VBF)
        z.append(twonll)
x = np.array(x)
y = np.array(y)
z = np.array(z)
z = z - np.min(z)

points = np.array([x, y, z])

# start interpolation
print("Interpolation")
x_int, y_int = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
z_int = griddata((x, y), z, (x_int, y_int), method="cubic")

fig, ax = plt.subplots()
ax, cm, pc = plot_as_heatmap(ax, x_int, y_int, z_int)
fig.colorbar(pc, ax=ax, label="-2$\Delta$lnL")
ax = plot_as_contour(ax, x_int, y_int, z_int, minimum, label="CL")
ax.set_xlabel("r_ggH")
ax.set_ylabel("r_VBF")
ax.legend(loc="upper right")
for ext in ["pdf", "png"]:
    fig.savefig(f"{output_dir}/scan_2D.{ext}")