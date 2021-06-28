# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -papermill
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Timing Functions
#
# This notebook has the results of timing various CSR functions.  It uses the data produced by the benchmark suite and saved in `bench.json`; to prepare that data, run:
#
#     python -m pytest --benchmark-only
#
# We use Jupytext to maintain the primary version of this notebook as a Python script.  To re-run the notebook and generate the `.ipynb` file:
#
#     jupytext -s Timings.py

# %%
import json
import numpy as np
import scipy.sparse as sps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# Let's load the benchmark data:

# %% tags=["parameters"]
bench_file = 'bench.json'

# %%
with open(bench_file, 'r') as jsf:
    data = json.load(jsf)
list(data.keys())


# %% [markdown]
# And define a function to get the various benchmark runs:

# %%
def get_runs(group, *params):
    marks = [b for b in data['benchmarks'] if b['group'] == group]
    ps = ['kernel'] + list(params)
    runs = {}
    for b in data['benchmarks']:
        if b['group'] != group:
            continue

        key = tuple(b['params'][p] for p in ps)
        runs[key] = pd.Series(b['stats']['data'], name='time')

    runs = pd.concat(runs, names=ps)
    runs = runs.reset_index(ps)
    runs = runs.reset_index(drop=True)
    return runs


# %%
get_runs('MultAB')

# %% [markdown]
# ## Matrix Multiplication
#
# Our first benchmark is a simple matrix multiplication.

# %%
mab = get_runs('MultAB')
mab['time'] *= 1000
mab.groupby('kernel')['time'].describe()

# %%
sns.catplot(data=mab, x='kernel', y='time', kind='bar')
plt.ylabel('ms / op')
plt.show()

# %% [markdown]
# And multiplying by the transpose:

# %%
mab = get_runs('MultABt')
mab['time'] *= 1000
mab.groupby('kernel')['time'].describe()

# %%
sns.catplot(data=mab, x='kernel', y='time', kind='bar')
plt.ylabel('ms / op')
plt.plot()

# %% [markdown]
# ### Sweep by Density
#
# We now measure sweeping a 100x100 square matrix multiply by increasing density.

# %%
dens = get_runs('MultAB-Density', 'density')
dens['time'] *= 1000

# %%
sns.lineplot(data=dens, x='density', y='time', hue='kernel')
plt.title('AB (100x100 square)')
plt.ylabel('ms/op')
plt.show()

# %% [markdown]
# And the transpose:

# %%
dens = get_runs('MultABt-Density', 'density')
dens['time'] *= 1000

# %%
sns.lineplot(data=dens, x='density', y='time', hue='kernel')
plt.title('AB\' (100x100 square)')
plt.ylabel('ms/op')
plt.show()

# %% [markdown]
# ### Sweep by Size
#
# We now measure sweeping a 10% square matrix multiply by increasing size.

# %%
sizes = get_runs('MultAB-Size', 'size')
sizes['time'] *= 1000

# %%
sns.lineplot(data=sizes, x='size', y='time', hue='kernel')
plt.title('AB (square, 10%)')
plt.ylabel('ms/op')
plt.show()

# %% [markdown]
# ## Matrix-Vector Multiplication
#
# Now we'll look at matrix/vector multiplication ($A\vec{x}$)

# %%
muax = get_runs('MultAx')
muax['time'] *= 1000
muax.groupby('kernel')['time'].describe()

# %%
sns.catplot(data=muax, x='kernel', y='time', kind='bar')
plt.ylabel('ms / op')
plt.show()

# %%
