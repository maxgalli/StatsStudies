from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.colors as colors

hep.style.use("CMS")


def plot_scan(arr, ax, color, var, xlims, label=None, y_text_pos=0.95):
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
 
    ax.set_xlabel(var)
    ax.set_ylabel("-2$\Delta$lnL")
    ax.set_ylim(0, 10)
    ax.set_xlim(*xlims)
    ax.axhline(1.0, color="k", linestyle="--")

    # add text with the best fit value and uncertainty
    nomstring = f"{minimum[0]:.3f}"
    up_unc = np.abs(points[0][0] - minimum[0])
    low_unc = np.abs(points[1][0] - minimum[0])
    upstring = f"{points[1][0]:.3f}"
    upstring = "{+" + f"{up_unc:.3f}" + "}"
    lowstring = f"{points[0][0]:.3f}"
    lowstring = "{-" + f"{low_unc:.3f}" + "}"
    fullstring = f"{var} = ${nomstring}^{upstring}_{lowstring}$"
    ax.text(0.05, y_text_pos, fullstring, transform=ax.transAxes, fontsize=16, va="top", ha="left")
    ax.legend(loc="upper right")
   
    return ax

def plot_as_data(data, nbins=100, normalize_to=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    if normalize_to is not None:
        # normalize to is the number of events of another dataset
        bins, edges = np.histogram(data, bins=nbins)
        bins = np.true_divide(bins, normalize_to)
        yerr = np.sqrt(bins) / np.sqrt(normalize_to)
    else:
        bins, edges = np.histogram(data, bins=nbins, density=True)
        yerr = np.sqrt(bins) / np.sqrt(len(data))
    centers = (edges[:-1] + edges[1:]) / 2
    ax.errorbar(centers, bins, yerr=yerr, fmt="o", color="black", **kwargs)
    # plot with correct error bars
    return ax

def save_image(name, outdir):
    for ext in ["png", "pdf"]:
        plt.savefig(f"{outdir}/{name}.{ext}")

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap

def custom_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    cmap = cmap.reversed()
    cmap = truncate_colormap(cmap, 0.3, 1.0, 1000)
    cmap_colors = cmap(np.linspace(minval, maxval, n))
    cmap_colors[-1] = np.array([1, 1, 1, 1])  # Set the highest level to white
    new_cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
    return new_cmap

def plot_as_heatmap(ax, x_int, y_int, z_int):
    colormap = custom_colormap("Purples")
    pc = ax.pcolormesh(
        x_int,
        y_int,
        z_int,
        vmin=0,
        vmax=10,
        cmap=colormap,
        shading="gouraud",
    )

    return ax, colormap, pc

def plot_as_contour(ax, x_int, y_int, z_int, minimum, color="k", label=None):
    cs = ax.contour(
        x_int,
        y_int,
        z_int,
        levels=[2.295748928898636, 6.180074306244173],
        colors=[color, color],
        linewidths=[2.2, 2.2],
        linestyles=["solid", "dashed"],
        #levels=[2.295748928898636, 6.180074306244173, 11.829158081900795, 19.333908611934685],
        #colors=[color, color, color, color],
        #linewidths=[2.2, 2.2, 2.2, 2.2],
        #linestyles=["solid", "dashed", "dashed", "dashed"],
    )
    # add labels
    try:
        levels = ["68%", "95%"]
        #levels = ["68%", "95%", "99.7%", "99.9%"]
        if label is not None:
            for i, cl in enumerate(levels):
                cs.collections[i].set_label(f"{label} {cl}")
    except IndexError as e:
        print(
            f"Could not add labels to contour plot because of error {e}."
        )
        if label is not None:
            cs.collections[0].set_label(f"{label} 68%")

    ax.plot(
        minimum[0],
        minimum[1],
        color=color,
        linestyle="",
        marker="o",
        label="Best fit",
    )

    return ax

@dataclass
class Likelihood:
    function: callable
    data: np.ndarray

    def __call__(self, *params):
        return np.prod(self.function(self.data, *params))


class NLL(Likelihood):
    def __call__(self, *params):
        return -np.sum([np.log(self.function(self.data, *params))])


class SumNLL:
    def __init__(self, *nlls):
        self.nlls = nlls

    def __call__(self, *params):
        return np.sum([nll(*params) for nll in self.nlls])