# scGACL
a generative adversarial network with  multi-scale contrastive learning for accurate  scRNA-seq imputation
## Overview
Single-cell RNA sequencing (scRNA-seq) is a powerful technology for studying cell-to-cell heterogeneity, but its application is hampered by dropout events, which introduce numerous non-biological zeros into the expression matrix. Therefore, accurate imputation of these zeros is crucial for downstream analyses. However, existing imputation methods often suffer from the over-smoothing problem, which leads to the loss of cell-to-cell heterogeneity in the imputed results. 

Here, we propose scGACL, a generative adversarial network (GAN) integrated with multi-scale contrastive learning. The GAN architecture drives the distribution of the imputed data to approximate that of the real data. To fundamentally address over-smoothing, the model incorporates a multi-scale contrastive learning mechanism: cell-level contrastive learning preserves fine-grained cell-to-cell heterogeneity, while cell-type-level contrastive learning maintains macroscopic biological variation across different cell groups. These mechanisms work in synergy to ensure imputation accuracy and  address the over-smoothing problem.
![GitHub图像](/scGACL_model_v2.png)

## Requirements
python==3.9.19

torch==2.2.2

scanpy==1.10.1

scikit-learn==1.3.0

matplotlib==3.8.4

numpy==1.26.4

pandas==2.2.1

scipy==1.13.0

h5py==3.11.0

tqdm=4.66.2

## Usage
### Run the demo
All the original scRNA-seq datasets can be downloaded ([Simulated 1-6](https://figshare.com/articles/software/scRNMF/23725986?file=41653401), [CellBench](https://github.com/LuyiTian/sc_mixology/tree/master), [GSE131907](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907), [GSM5436518](https://db.cngb.org/cdcp/dataset/SCDS0000567/), [Deng](https://figshare.com/articles/software/scRNMF/23725986?file=41653401)).

We can quickly start scGACL：
1. preprocess scRNA-seq data:

```python 
python trainer/datasets.py  # generate preprocess.h5ad
```

2. impute scRNA-seq data:
```python
python main.py # generate result.csv
```

### Input file
scGACL requires a gene expression matrix as input. The data should be preprocessed to generate an h5ad file with cells as rows and genes as columns. If cell type labels are available, please provide them during preprocessing.
These labels should be stored in the obs field of the h5ad file, with the column name: cell_type. The resulting preprocessed h5ad file will serve as the input for the imputation stage.

### Output file
The output of scGACL includes the final imputed expression matrix, saved as result.csv, and the trained model files stored in the net_state_dict/ folder.

### Hyperparameters
Configuration files containing scGACL's hyperparameters can be found in the trainer/ folder. Key hyperparameters include:
|  `Parameter`   | Description  |
|  :----  | :----  |
| `CUDA` | Whether to use GPU for training. |
| `GENENUM` | Number of genes to be imputed. |
| `Z_SIZE`  | Dimensionality of the latent space in VaDE. |
| `N_CLASS` | If cell type labels are available in the scRNA-seq dataset, this parameter specifies the number of distinct cell types. |
| `N_CLUSTER` | If no cell type labels are provided, the model performs preliminary clustering on the dataset, and this parameter defines the number of clusters. |
| `VAE_EPOCH` | Number of epochs for Phase-1 training of VaDE. |
| `VAE_LR` | Learning rate for Phase-1 training of VaDE. |
| `VAE_BATCH_SIZE` | Batch size for Phase-1 training of VaDE. |
| `EPOCH` | Number of training epochs for Phase-2 joint training of VaDE and the multi-task discriminator. |
| `LR` | Learning rate for Phase-2 joint training of VaDE and the multi-task discriminator. |
| `BATCH_SIZE` | Batch size for Phase-2 joint training of VaDE and the multi-task discriminator. |
| `LAMBDA_G_ADV` | Weight coefficient for the adversarial loss. |
| `LAMBDA_SIMCL` | Weight coefficient for the cell-level contrastive loss. |
| `LAMBDA_SUPCL` | Weight coefficient for the cell-type-level contrastive loss. |
| `TEMPERATURE` | The temperature parameter used in contrastive learning. |


