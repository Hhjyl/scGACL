# scGACL
a generative adversarial network with  multi-scale contrastive learning for accurate  scRNA-seq imputation
## Overview
Single-cell RNA sequencing (scRNA-seq) is a powerful technology for studying cell-to-cell heterogeneity, but its application is hampered by dropout events, which introduce numerous non-biological zeros into the expression matrix. Therefore, accurate imputation of these zeros is crucial for downstream analyses. However, existing imputation methods often suffer from the over-smoothing problem, which leads to the loss of cell-to-cell heterogeneity in the imputed results. 

Here, we propose scGACL, a generative adversarial network (GAN) integrated with multi-scale contrastive learning. The GAN architecture drives the distribution of the imputed data to approximate that of the real data. To fundamentally address over-smoothing, the model incorporates a multi-scale contrastive learning mechanism: cell-level contrastive learning preserves fine-grained cell-to-cell heterogeneity, while cell-type-level contrastive learning maintains macroscopic biological variation across different cell groups. These mechanisms work in synergy to ensure imputation accuracy and  address the over-smoothing problem.

## Requirements
