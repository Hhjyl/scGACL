import copy
import math

from sklearn.cluster import KMeans

from trainer.config import cfg
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import scanpy as sc
import torch


class MyDataset(Dataset):
    """Operations with the datasets."""

    def __init__(self, ad, use_label=True):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = np.array(ad.X)  # anndata格式存表达值和细胞类型
        self.data = torch.tensor(self.data, dtype=torch.float)
        if use_label:
            self.data_cls = pd.Categorical(ad.obs['cell_type']).codes  #
        else:
            sc.pp.pca(ad)
            sc.pp.neighbors(ad)
            X_pca = ad.obsm['X_pca']

            kmeans = KMeans(n_clusters=cfg.N_CLUSTER, random_state=0)
            clusters = kmeans.fit_predict(X_pca)
            ad.obs['kmeans'] = clusters.astype(str)

            self.data_cls = pd.Categorical(ad.obs['kmeans']).codes
        self.data_cls = torch.tensor(self.data_cls, dtype=torch.long)

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.data_cls[idx]
        return data, label


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        # sc.pp.normalize_total(adata, target_sum=1e4)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    if adata.n_vars > 5000:
        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        adata = adata[:, adata.var['highly_variable'] == True].copy()

    return adata


if __name__ == '__main__':
    drop_expression_path = r'F:/pythonProject/scGACL/dataset/sim.groups3_counts.csv'
    celltype_path = r'F:/pythonProject/scGACL/dataset/sim.groups3_groups.csv'
    drop_rna_ad = sc.read(drop_expression_path).T
    drop_rna_ad = normalize(drop_rna_ad)
    print(drop_rna_ad)
    print(drop_rna_ad.to_df())
    celltype_df = pd.read_csv(celltype_path, index_col=0)
    print(celltype_df)
    new_index = []
    for index in celltype_df.index:
        index = str(index)
        index = 'Cell' + index
        new_index.append(index)
    celltype_df.index = new_index
    cells = drop_rna_ad.obs_names.tolist()
    celltype_df = celltype_df.loc[cells, :]
    drop_rna_ad.obs['cell_type'] = celltype_df.loc[:, 'cell_type']

    print(drop_rna_ad)
    print(drop_rna_ad.obs)
    sc.write(r'F:/pythonProject/scGACL/dataset/sim.groups1_preprocess.h5ad', drop_rna_ad)
# if __name__ == '__main__':
#     drop_expression_path = r'F:\simulate_dataset\dataset1\counts.csv'
#     celltype_path = r'F:\simulate_dataset\dataset1\groups.csv'
#     drop_rna_ad = sc.read(drop_expression_path).T
#     drop_rna_ad = normalize(drop_rna_ad)
#     print(drop_rna_ad)
#     print(drop_rna_ad.to_df())
#     celltype_df = pd.read_csv(celltype_path, index_col=0)
#     print(celltype_df)
#
#     cells = drop_rna_ad.obs_names.tolist()
#     celltype_df = celltype_df.loc[cells, :]
#     drop_rna_ad.obs['cell_type'] = celltype_df.loc[:, 'Group']
#
#     print(drop_rna_ad)
#     print(drop_rna_ad.obs)


