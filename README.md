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
The scRNA-seq datasets can be downloaded

