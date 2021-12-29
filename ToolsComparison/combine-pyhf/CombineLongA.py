import json

import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhf.contrib.viz import brazil

import logging
logger = logging.getLogger("")

def setup_logging(level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(levelname)s - %(filename)s:%(lineno)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return log

log = setup_logging(logging.DEBUG)


# Main
spec = json.load(open("datacard_part1.json"))
workspace = pyhf.Workspace(spec)
model = workspace.model()

observations = [ob[0] for ob in list(workspace.observations.values())] + model.config.auxdata

pyhf.infer.mle.fit(data=observations, pdf=model)

CLs_obs, CLs_exp = pyhf.infer.hypotest(
    1.0,  # null hypothesis
    observations,
    model,
    test_stat="qtilde",
    return_expected_set=True,
)
print(f"      Observed CLs: {CLs_obs:.4f}")
for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
    print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.4f}")

poi_values = np.linspace(0.1, 5, 50)
obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upperlimit(
    observations, model, poi_values, level=0.05, return_results=True
)
print(f"Upper limit (obs): μ = {obs_limit:.4f}")
print(f"Upper limit (exp): μ = {exp_limits[2]:.4f}")

#fig, ax = plt.subplots()
#fig.set_size_inches(10.5, 7)
#ax.set_title("Hypothesis Tests")

#artists = brazil.plot_results(poi_values, results, ax=ax)