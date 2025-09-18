# SpaGIM: Spatial Gene Imputation Methods
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

Install Harmony via pip:
```bash
pip install harmonypy
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

# Run gene expression imputation using selected method
adata_impute = models.forward(
    adata_sc,      # Single-cell reference dataset (AnnData object)
    adata_sp,      # Spatial transcriptomics dataset (AnnData object)

    # Select one of the imputation methods:
    # 'Pearson', 'Spearman', 'Cosine'       → Similarity-based (correlation/distance)
    # 'KNN'                                 → k-nearest neighbors averaging
    # 'SpaGE', 'Harmony'                    → Integration-based methods
    # 'Tangram'                             → Deep learning-based spatial mapping
    # 'ENVI', 'GIMVI'                       → Probabilistic deep generative models
    method='Pearson',

    # Number of data splits (only used for: 'Tangram', 'Pearson', 'Spearman', 'Cosine')
    # Use >1 for memory-efficient chunk-wise computation
    n_split=1,

    # Batch size for gradient-based methods (used only by 'GIMVI')
    # Controls memory usage and convergence speed during training
    batch_size=1024,

    # Number of nearest neighbors to use for similarity-based methods
    # Applicable for: 'KNN', 'Pearson', 'Spearman', 'Cosine', 'Harmony'
    k=100
)
```
