# StatsStudies
Statistics studies and exercises based on Mauro's lectures

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

## Procedure

The idea consists in running the Combine long exercise and implement every step also with a different Python package (zfit, pyhf, even something in Julia) to show how easier and clear it is, understand what happens and compare the performance.