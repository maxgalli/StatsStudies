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