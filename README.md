# StatsStudies
Statistics studies and exercises based on Mauro's lectures and HEP stuff in
general.

## Setup

```
$ conda env create -f environment.yml
```
or
```
$ mamba env create -f environment.yml
```

**N.B.**: zfit might cause some problems due to dependency on tensorflow. If so:
```
$ conda install -c anaconda pip
$ pip install zfit --upgrade
```

### For ipywidgets to work

```
mamba install -c conda-forge ipywidgets
mamba install -c conda-forge ipympl
mamba install nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension
```

## Procedure

The idea consists in running the Combine long exercise and implement every step also with a different Python package (zfit, pyhf, even something in Julia) to show how easier and clear it is, understand what happens and compare the performance.
