---
jupytext:
  cell_metadata_filter: tags,-papermill
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Timing Functions

This notebook has the results of timing various CSR functions.  It uses the data produced by the benchmark suite and saved in `bench.json`; to prepare that data, run:

    python -m pytest --benchmark-only

We use Jupytext to maintain the primary version of this notebook as a Markdown file.  To re-run the notebook and generate the `.ipynb` file:

    jupytext -s --execute Timings.md

```{code-cell} ipython3
import json
import numpy as np
import scipy.sparse as sps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Let's load the benchmark data:

```{code-cell} ipython3
:tags: [parameters]

bench_file = 'bench.json'
```

```{code-cell} ipython3
with open(bench_file, 'r') as jsf:
    data = json.load(jsf)
list(data.keys())
```

And define a function to get the various benchmark runs:

```{code-cell} ipython3
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
```

```{code-cell} ipython3
get_runs('MultAB')
```

## Matrix Multiplication

Our first benchmark is a simple matrix multiplication.

```{code-cell} ipython3
mab = get_runs('MultAB')
mab['time'] *= 1000
mab.groupby('kernel')['time'].describe()
```

```{code-cell} ipython3
sns.catplot(data=mab, x='kernel', y='time', kind='bar')
plt.ylabel('ms / op')
plt.show()
```

And multiplying by the transpose:

```{code-cell} ipython3
mab = get_runs('MultABt')
mab['time'] *= 1000
mab.groupby('kernel')['time'].describe()
```

```{code-cell} ipython3
sns.catplot(data=mab, x='kernel', y='time', kind='bar')
plt.ylabel('ms / op')
plt.plot()
```

### Sweep by Density

We now measure sweeping a 100x100 square matrix multiply by increasing density.

```{code-cell} ipython3
dens = get_runs('MultAB-Density', 'density')
dens['time'] *= 1000
```

```{code-cell} ipython3
sns.lineplot(data=dens, x='density', y='time', hue='kernel')
plt.title('AB (100x100 square)')
plt.ylabel('ms/op')
plt.show()
```

And the transpose:

```{code-cell} ipython3
dens = get_runs('MultABt-Density', 'density')
dens['time'] *= 1000
```

```{code-cell} ipython3
sns.lineplot(data=dens, x='density', y='time', hue='kernel')
plt.title('AB\' (100x100 square)')
plt.ylabel('ms/op')
plt.show()
```

### Sweep by Size

We now measure sweeping a 10% square matrix multiply by increasing size.

```{code-cell} ipython3
sizes = get_runs('MultAB-Size', 'size')
sizes['time'] *= 1000
```

```{code-cell} ipython3
sns.lineplot(data=sizes, x='size', y='time', hue='kernel')
plt.title('AB (square, 10%)')
plt.ylabel('ms/op')
plt.show()
```

## Matrix-Vector Multiplication

Now we'll look at matrix/vector multiplication ($A\vec{x}$)

```{code-cell} ipython3
muax = get_runs('MultAx')
muax['time'] *= 1000
muax.groupby('kernel')['time'].describe()
```

```{code-cell} ipython3
sns.catplot(data=muax, x='kernel', y='time', kind='bar')
plt.ylabel('ms / op')
plt.show()
```

```{code-cell} ipython3

```
