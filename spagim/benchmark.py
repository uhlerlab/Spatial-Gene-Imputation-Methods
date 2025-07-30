
import random
import time
import numpy as np
import anndata
import torch
import scanpy as sc
import pandas as pd
import tangram as tg
from scvi.external import GIMVI
from scipy.stats import rankdata
import scenvi
from spagim.main import SpaGE
from sklearn.neighbors import NearestNeighbors
import scanpy.external as sce
from scipy.spatial import KDTree

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
class Imputation:
    '''
    Benchmark Imputation
    '''
    def __init__(self, device='cpu', seed=0):
        self.device = device
        set_seed(seed)
        
    def split_batches(self, data, n_split):
        '''
        Split data into batches
        '''
        batch_size = int(np.ceil(data.shape[0] / n_split))
        return [data[i * batch_size:(i + 1) * batch_size] for i in range(n_split)]
    
    def split_adata(self, adata, k):
        """
        Split AnnData object into k subsets
        """
        num_cells = adata.shape[0]
        shuffled_indices = np.random.permutation(num_cells)
        subset_size = num_cells // k
        adata_list = []

        for i in range(k):
            if i == k - 1: 
                indices = shuffled_indices[i * subset_size:]
            else:
                indices = shuffled_indices[i * subset_size:(i + 1) * subset_size]

            adata_subset = adata[indices, :].copy()
            adata_list.append(adata_subset)

        return adata_list

    def tangram(self, adata_sc, adata_sp, n_split=1):
        '''
        Run Tangram imputation
        '''
    
        tg.pp_adatas(adata_sc, adata_sp, genes=None)
        
        ## split spatial data into n_split batches
        if n_split > 1:
            adata_pred_list = []
            adata_sp_list = self.split_adata(adata_sp, n_split)
            for i in range(n_split):
                time1 = time.time()
                adata_map = tg.map_cells_to_space(
                    adata_sc=adata_sc,
                    adata_sp=adata_sp_list[i],
                    device=self.device,
                )
                adata_pred = tg.project_genes(adata_map, adata_sc)
                adata_pred_list.append(adata_pred)
                time2 = time.time()
                print('Time cost minutes: ', (time2-time1)/60)
            # reindex adata_sp_list[i] to adata_pred
            adata_pred_combined = anndata.concat(adata_pred_list, axis=0)
            adata_pred = adata_pred_combined[adata_sp.obs_names].copy()
            
        else:
            adata_map = tg.map_cells_to_space(
                adata_sc=adata_sc,
                adata_sp=adata_sp,
                device=self.device,
            )
            adata_pred = tg.project_genes(adata_map, adata_sc)
        return adata_pred
    
    def gimvi(self, adata_sc, adata_sp, batch_size=1024):
        '''
        Run GIMVI imputation
        '''
        GIMVI.setup_anndata(adata_sp)
        GIMVI.setup_anndata(adata_sc)
        model = GIMVI(adata_sc, adata_sp)
        model.train(200, batch_size=batch_size)  
        _, imputation = model.get_imputed_values(normalized=False)
        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        
        return adata_pred
    
    def spearman(self, adata_sc, adata_sp, n_split=50, top_k=100):
        '''
        Calculate Spearman correlation with batch processing on GPU
        '''
        
        adata_sc_train = adata_sc[:, adata_sp.var_names]
        data_st = adata_sp.X
        data_sc = adata_sc_train.X 
        
        rank_st = np.apply_along_axis(rankdata, 1, data_st)
        rank_sc = np.apply_along_axis(rankdata, 1, data_sc)

        rank_sc_tensor = torch.tensor(rank_sc, device=self.device, dtype=torch.float32)
        
        top_k = top_k
        imputation = torch.zeros(adata_sp.shape[0], adata_sc.shape[1], device='cpu') 

        rank_st_batches = self.split_batches(rank_st, n_split)
        
        start_idx = 0  
        for i, rank_st_batch in enumerate(rank_st_batches):
            print('Processing batch', i)
            
            batch_size = rank_st_batch.shape[0]
            
            rank_st_tensor = torch.tensor(rank_st_batch, device=self.device, dtype=torch.float32)
            
            mean_rank_st = torch.mean(rank_st_tensor, dim=1, keepdim=True)
            mean_rank_sc = torch.mean(rank_sc_tensor, dim=1, keepdim=True)
            rank_st_centered = rank_st_tensor - mean_rank_st
            rank_sc_centered = rank_sc_tensor - mean_rank_sc
            
            cov_matrix = torch.matmul(rank_st_centered, rank_sc_centered.t())
            
            norm_st = torch.sqrt(torch.sum(rank_st_centered ** 2, dim=1, keepdim=True))
            norm_sc = torch.sqrt(torch.sum(rank_sc_centered ** 2, dim=1, keepdim=True))
            norm_matrix = torch.matmul(norm_st, norm_sc.t())
            norm_matrix = torch.clamp(norm_matrix, min=1e-8)
            
            spearman_corr_batch = cov_matrix / norm_matrix
            
            top_k_values, top_k_indices = torch.topk(spearman_corr_batch, top_k, dim=1)
            
            adata_sc_tensor = torch.tensor(adata_sc.X, device=self.device, dtype=torch.float32)
            selected_data_sc = adata_sc_tensor[top_k_indices]
            
            weighted_sum = torch.einsum('ij,ijk->ik', top_k_values, selected_data_sc)
            
            imputation[start_idx:start_idx + batch_size] = \
                (weighted_sum / torch.sum(top_k_values, dim=1, keepdim=True)).cpu()
            
            start_idx += batch_size

        imputation = imputation.numpy()

        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        
        return adata_pred

    def cosine_similarity_with_covariance(self, adata_sc, adata_sp, n_split=50, top_k=100, radius=100):
        '''
        Calculate cosine similarity, incorporating gene-gene covariance based on spatial proximity.
        '''
        
        adata_sc_train = adata_sc[:, adata_sp.var_names]
        data_st = adata_sp.X
        data_sc = adata_sc_train.X

        data_sc_tensor = torch.tensor(adata_sc.X, device=self.device, dtype=torch.float32)
        data_sc_train_tensor = torch.tensor(data_sc, device=self.device, dtype=torch.float32)

        # Initialize imputation tensor
        imputation = torch.zeros(adata_sp.shape[0], adata_sc.shape[1], device='cpu')

        # Get spatial coordinates of cells in adata_sp
        spatial_coords = adata_sp.obsm['spatial']

        # Create KDTree for fast spatial neighbor lookup (using all spatial coordinates)
        tree = KDTree(spatial_coords)

        # Find all neighbors for all cells at once (this is not batch-based)
        all_neighbors_indices = tree.query_ball_point(spatial_coords, radius)

        # Split data_st into batches for processing, but not for KNN
        data_st_batches = self.split_batches(data_st, n_split)

        # Normalize data_sc_tensor once for all batches (KNN doesn't depend on batch)
        norm_sc = torch.clamp(torch.norm(data_sc_train_tensor, dim=1, keepdim=True), min=1e-8)
        data_sc_normalized = data_sc_train_tensor / norm_sc

        start_idx = 0
        for batch_idx, data_st_batch in enumerate(data_st_batches):
            print(f'Processing batch {batch_idx}')

            batch_size = data_st_batch.shape[0]
            data_st_tensor = torch.tensor(data_st_batch, device=self.device, dtype=torch.float32)

            norm_st = torch.clamp(torch.norm(data_st_tensor, dim=1, keepdim=True), min=1e-8)
            data_st_normalized = data_st_tensor / norm_st

            # Calculate cosine similarity between the current batch and all sc cells (KNN across all data)
            cosine_sim_batch = torch.matmul(data_st_normalized, data_sc_normalized.t())

            # Use precomputed neighbors for the cells in the current batch
            neighbors_indices = all_neighbors_indices[start_idx:start_idx + batch_size]

            # Prepare tensor to hold covariance similarities for the batch
            cov_sim_batch = torch.zeros_like(cosine_sim_batch, device=self.device)

            # Batch calculate gene-gene covariance matrix similarity
            for i, neighbor_idx in enumerate(neighbors_indices):
                # Skip cells with no neighbors
                if len(neighbor_idx) < 2:
                    continue

                # Get neighbor gene expression data for all neighbors of cell i
                local_gene_data = torch.tensor(adata_sp.X[neighbor_idx], device=self.device, dtype=torch.float32)

                # Check for NaN or inf values in the data
                if torch.isnan(local_gene_data).any() or torch.isinf(local_gene_data).any():
                    print(f"Skipping cell {i} due to NaN or inf in local gene data.")
                    continue

                # Compute gene-gene covariance matrix using torch
                local_mean = torch.mean(local_gene_data, dim=0)
                local_centered = local_gene_data - local_mean

                # Check for zero variance in the data to prevent issues with covariance calculation
                if torch.all(local_centered == 0):
                    print(f"Skipping cell {i} due to zero variance in local gene data.")
                    continue

                # Compute covariance matrix
                gene_cov_matrix = torch.matmul(local_centered.T, local_centered) / (local_centered.shape[0] - 1)

                # Flatten and normalize the covariance matrix
                cov_vector = gene_cov_matrix.flatten()
                cov_norm = torch.clamp(torch.norm(cov_vector, keepdim=True), min=1e-8)
                cov_vector_normalized = cov_vector / cov_norm

                # Compute covariance similarity between cell i and all other cells (batch processing)
                cov_sim_batch[i] = torch.matmul(cov_vector_normalized, cov_vector_normalized.T)

            # Combine cosine similarity and covariance similarity
            combined_similarity = cosine_sim_batch + cov_sim_batch

            # Get top_k based on the combined similarity (KNN across all data)
            top_k_values, top_k_indices = torch.topk(combined_similarity, top_k, dim=1)

            # From data_sc_tensor, select the gene expression data of the top_k similar cells
            selected_data_sc = data_sc_tensor[top_k_indices]

            # Calculate weighted sum
            weighted_sum = torch.einsum('ij,ijk->ik', top_k_values, selected_data_sc)

            # Compute weighted average and store it in imputation tensor
            imputation[start_idx:start_idx + batch_size] = (weighted_sum / torch.sum(top_k_values, dim=1, keepdim=True)).cpu()

            # Update batch start index
            start_idx += batch_size

        # Convert imputation tensor to numpy and create AnnData object
        imputation = imputation.numpy()
        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        print(adata_pred.X.max())

        return adata_pred
    
    def cosine_similarity(self, adata_sc, adata_sp, n_split=50, top_k=100):
        '''
        Calculate cosine similarity and imputation directly inside the batch loop
        '''
        
        adata_sc_train = adata_sc[:, adata_sp.var_names]
        data_st = adata_sp.X
        data_sc = adata_sc_train.X
        
        data_sc_tensor = torch.tensor(adata_sc.X, device=self.device, dtype=torch.float32)  
        data_sc_train_tensor = torch.tensor(data_sc, device=self.device, dtype=torch.float32)  
        
        top_k = top_k
        imputation = torch.zeros(adata_sp.shape[0], adata_sc.shape[1], device='cpu')  
        
        data_st_batches = self.split_batches(data_st, n_split)
        
        start_idx = 0  
        for batch_idx, data_st_batch in enumerate(data_st_batches):
            print(f'Processing batch {batch_idx}')
            
            batch_size = data_st_batch.shape[0]
            
            data_st_tensor = torch.tensor(data_st_batch, device=self.device, dtype=torch.float32)
            
            norm_st = torch.clamp(torch.norm(data_st_tensor, dim=1, keepdim=True), min=1e-8)
            norm_sc = torch.clamp(torch.norm(data_sc_train_tensor, dim=1, keepdim=True), min=1e-8)
            
            data_st_normalized = data_st_tensor / norm_st
            data_sc_normalized = data_sc_train_tensor / norm_sc
            
            cosine_sim_batch = torch.matmul(data_st_normalized, data_sc_normalized.t())
            
            top_k_values, top_k_indices = torch.topk(cosine_sim_batch, top_k, dim=1)
            
            selected_data_sc = data_sc_tensor[top_k_indices]
            
            weighted_sum = torch.einsum('ij,ijk->ik', top_k_values, selected_data_sc)
            
            imputation[start_idx:start_idx + batch_size] = \
                (weighted_sum / torch.sum(top_k_values, dim=1, keepdim=True)).cpu()
            
            start_idx += batch_size
        
        imputation = imputation.numpy()
        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        
        return adata_pred
    
    def pearson(self, adata_sc, adata_sp, n_split=50, top_k=100):
        '''
        Calculate Pearson correlation with batch processing and use top 100 highest correlations for imputation
        '''
        
        adata_sc_train = adata_sc[:, adata_sp.var_names]
        data_st = adata_sp.X
        data_sc = adata_sc_train.X 
        
        data_sc_train_tensor = torch.tensor(data_sc, device=self.device, dtype=torch.float32)
        data_sc_tensor = torch.tensor(adata_sc.X, device=self.device, dtype=torch.float32)
        
        top_k = top_k
        imputation = torch.zeros(adata_sp.shape[0], adata_sc.shape[1], device='cpu')  
        
        data_st_batches = self.split_batches(data_st, n_split)
        
        start_idx = 0  
        for i, data_st_batch in enumerate(data_st_batches):
            print('Processing batch', i)
            
            batch_size = data_st_batch.shape[0]
            
            data_st_tensor = torch.tensor(data_st_batch, device=self.device, dtype=torch.float32)
            
            mean_data_st = torch.mean(data_st_tensor, dim=1, keepdim=True)
            mean_data_sc = torch.mean(data_sc_train_tensor, dim=1, keepdim=True)
            data_st_centered = data_st_tensor - mean_data_st
            data_sc_centered = data_sc_train_tensor - mean_data_sc
            
            cov_matrix = torch.matmul(data_st_centered, data_sc_centered.t())
            
            norm_st = torch.sqrt(torch.sum(data_st_centered ** 2, dim=1, keepdim=True))
            norm_sc = torch.sqrt(torch.sum(data_sc_centered ** 2, dim=1, keepdim=True))
            norm_matrix = torch.matmul(norm_st, norm_sc.t())
            norm_matrix = torch.clamp(norm_matrix, min=1e-8)
            
            pearson_corr_batch = cov_matrix / norm_matrix
            
            top_k_values, top_k_indices = torch.topk(pearson_corr_batch, top_k, dim=1)
            
            selected_data_sc = data_sc_tensor[top_k_indices]
            
            weighted_sum = torch.einsum('ij,ijk->ik', top_k_values, selected_data_sc)
            imputation[start_idx:start_idx + batch_size] = \
                (weighted_sum / torch.sum(top_k_values, dim=1, keepdim=True)).cpu()
            
            start_idx += batch_size
        
        imputation = imputation.numpy()

        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        
        return adata_pred

    def knn(self, adata_sc, adata_sp, k=100):
        '''
        Run KNN imputation
        '''
        
        adata_sc_train = adata_sc[:, adata_sp.var_names]
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(adata_sc_train.X)
        _, ind = knn.kneighbors(adata_sp.X)
        print('Imputing...')
        
        imputation = np.zeros(adata_sc.X.shape)
        neighbors_matrix = adata_sc.X[ind]
        imputation = np.mean(neighbors_matrix, axis=1)
            
        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        
        return adata_pred
    
    def envi(self, adata_sc, adata_sp, cov_coeff=1):
        '''
        Run ENVI imputation
        '''
        
        envi_model = scenvi.ENVI(spatial_data = adata_sp, sc_data = adata_sc, num_HVG=adata_sc.X.shape[1], cov_coeff=cov_coeff)
        envi_model.train()
        envi_model.impute_genes()
        envi_model.infer_niche_covet()

        imputation = envi_model.spatial_data.obsm['imputation']
        imputation = imputation.reindex(columns=adata_sc.var_names)
        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        return adata_pred

    def harmony(self, adata_sc, adata_sp, k=100):
        '''
        Run Harmony imputation
        '''
        adata_sc_train = adata_sc[:, adata_sp.var_names]
        adata_sc_train.obs['batch'] = 'scRNA'
        adata_sp.obs['batch'] = 'spatial'
        adata = anndata.concat([adata_sc_train, adata_sp])
        
        sc.pp.pca(adata)
        sce.pp.harmony_integrate(adata, 'batch')
        
        adata_sc_train = adata[adata.obs['batch'] == 'scRNA'].copy()
        adata_sp = adata[adata.obs['batch'] == 'spatial'].copy()
        
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(adata_sc_train.obsm['X_pca_harmony'])
        _, ind = knn.kneighbors(adata_sp.obsm['X_pca_harmony'])
        print('Imputing...')
        
        imputation = np.zeros(adata_sc.X.shape)
        neighbors_matrix = adata_sc.X[ind]
        imputation = np.mean(neighbors_matrix, axis=1)
            
        adata_pred = anndata.AnnData(X=imputation, obs=adata_sp.obs, var=adata_sc.var)
        
        return adata_pred
    
    def spage(self, adata_sc, adata_sp):
        '''
        Run spage imputation
        '''

        df_sp = adata_sp.to_df().T
        df_sc = adata_sc.to_df().T
        
        # find missing genes
        missing_genes = list(set(df_sc.index.tolist()) - set(df_sp.index.tolist()))

        Imp_Genes = SpaGE(df_sp.T,df_sc.T,n_pv=30,
                           genes_to_predict = missing_genes)
        
        gene_var = pd.Index(missing_genes)
        adata_pred = anndata.AnnData(X=Imp_Genes, obs=adata_sp.obs, var=pd.DataFrame(index=gene_var))
        
        return adata_pred            