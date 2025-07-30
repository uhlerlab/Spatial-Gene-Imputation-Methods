# Run Gene Imputation Methods
[![PyPI version](https://img.shields.io/pypi/v/spagim)](https://pypi.org/project/spagim/)
[![Downloads](https://static.pepy.tech/badge/spagim)](https://pepy.tech/project/spagim)

**spagim** is a package for gene imputation of spatial transcriptomics data from single-cell RNA sequencing data. It provides a unified interface for various imputation methods, making it easy to apply them to your data.

## Installation
Create a conda environment

```bash
conda create -n spagim-env python=3.9
conda activate spagim-env
```
Install JAX via pip:

```bash
pip install -U "jax[cuda12]"
```

Install **spagim** via pip:

```bash
pip install spagim
```

## Usage

Here is a simple example of how to use **spagim**:

```python
import spagim.benchmark as benchmark
import scanpy as sc
import torch

# Load your spatial transcriptomics data
adata_sp = sc.read_h5ad("path/to/your/spatial_data")

# Load your single-cell RNA sequencing data
adata_sc = sc.read_h5ad("path/to/your/single_cell_data")

# Initialize the imputation model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = benchmark.Imputation(device=device, seed=42)

# Run imputation
adata_impute = models.forward(
    adata_sc,
    adata_sp,
    # choose your method among ['Pearson','Spearman','Cosine','KNN','SpaGE','Harmony','Tangram','ENVI','GIMVI']
    method='Pearson',
    # n_split is the number of splits for the data for ['Tangram','Pearson','Spearman','Cosine'] only
    n_split=1,
    # batch_size is the number of samples per gradient update, for 'GIMVI' only
    batch_size=1024,
    # k is the number of nearest neighbors for ['KNN','Pearson','Spearman','Cosine','Harmony'] only
    k=100
)
```
