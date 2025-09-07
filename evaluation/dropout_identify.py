import copy

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import gamma, norm
from scipy.optimize import root
from scipy.special import digamma
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import sys
from trainer.config import cfg
import scipy.stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

#SCDD代码  F:\pythonProject\SCDD-master\utils\prepropcess.py
def calculate_dropout_metrics(true_dropout, predicted_dropout):
    # Flatten the matrices to 1D arrays
    true_flat = true_dropout.flatten()
    pred_flat = predicted_dropout.flatten()

    # Calculate metrics
    accuracy = accuracy_score(true_flat, pred_flat)
    precision = precision_score(true_flat, pred_flat)
    recall = recall_score(true_flat, pred_flat)
    f1 = f1_score(true_flat, pred_flat)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# calculate the q()
def calculate_weight(xdata, params):
    # 计算每个数据点属于伽马分布和高斯分布的权重（即每个数据点属于这两个分布的概率）
    rate, alpha, scale = params[0], params[1], 1 / params[2]
    mu, std = params[3], params[4]
    pz1 = params[0] * gamma.pdf(xdata, a=alpha, scale=scale)  # scale = 1 / beta
    pz2 = (1 - params[0]) * norm.pdf(xdata, mu, std)
    pz = pz1 / (pz1 + pz2)
    pz[pz1 == 0] = 0
    wt = [pz, 1 - pz]
    # wt[0][i]: 第 i 个数据点属于伽马分布的责任权重。
    # wt[1][i]: 第 i 个数据点属于高斯分布的责任权重。
    return wt  # Represents the weight of gamma distribution and normal distribution


# Update parameters of gamma function
def update_gamma_pars(xdata, wt):
    xdata = np.array(xdata)
    wt = np.array(wt)
    tp_s = np.sum(wt)
    tp_t = np.sum(wt * xdata)
    tp_u = sum(wt * np.log(xdata))
    tp_v = -tp_u / tp_s - np.log(tp_s / tp_t)
    alpha = 0
    if tp_v <= 0:
        alpha = 20
    else:
        alpha0 = (3 - tp_v + np.sqrt((tp_v - 3) ** 2 + 24 * tp_v)) / 12 / tp_v
        if alpha0 >= 20:
            alpha = 20
        else:
            alpha = root(lambda alpha: np.log(alpha) - digamma(alpha) - tp_v, np.array([0.9, 1.1]) * alpha0)
            alpha = alpha.x[0]
    # need to solve log(x) - digamma(x) = tp_v
    # We use this approximation to compute the initial value
    beta = tp_s / tp_t * alpha
    return alpha, beta


# Update the model with new parameters
def dmix(xdata, params):
    rate, alpha, beta, mu, std = params
    return rate * gamma.pdf(xdata, a=alpha, scale=(1 / beta)) + (1 - rate) * norm.pdf(xdata, mu, std)


# Returns the parameters of the mix model
def get_mix_internal(xdata, point):
    # 使用期望最大化算法计算某个基因的模型参数
    # xdata是当前基因在所有样本的表达值
    # rate是公式的lambda
    # "rate", "alpha", "beta", "mu", "sigma"
    params = [0, 0, 0, 0, 0]
    params[0] = np.sum(xdata == point) / len(xdata)  # 计算当前基因表达为0的比例
    if params[0] > 0.95:  # When a gene has 95% dropouts, we consider it invalid
        params = np.repeat(np.nan, 5)
        return params
    if params[0] == 0:
        params[0] = 0.01  # init rate = 0.01
    # 初始化伽马分布的参数 alpha和 beta
    params[1], params[2] = 1.5, 1  # α，β = 0.5, 1
    # 筛选出所有大于 point 的表达值，用来估算高斯分布的参数。低表达的部分（小于或者等于阈值 point）将被忽略。
    xdata_rm = xdata[xdata > point]
    params[3], params[4] = np.mean(xdata_rm), np.std(xdata_rm)  # μ， σ
    if params[4] == 0:  # the variance is 0
        params[4] = 0.01
    eps = 10  # loss
    iter = 0  # iterations
    loglik_old = 0

    # The EM algorithm is continuously executed until the loss is reduced to less than 0.5
    # 使用期望最大化（EM）算法进行循环，直到误差小于 0.5 或迭代次数超过 100则跳出循环
    while eps > 0.5:
        # 获得2个分布的权重, 每列的值表示一个数据点属于两种分布的概率，列向量之和为 1
        wt = calculate_weight(xdata, params)
        # 计算伽马分布（wt[0]）和高斯分布（wt[1]）在所有数据点上的权重总和，结果是一个长度为 2 的数组,tp_sum[0] 是伽马分布在所有数据点上的总权重（即所有数据点属于伽马分布的概率之和）。
        tp_sum = np.sum(wt, axis=1)

        # --- M 步：更新参数 ---
        rate = tp_sum[0] / len(wt[0])
        # 所有数据点的加权平均，其中权重由 wt[1]（每个数据点属于高斯分布的责任）给出
        mu = np.sum(wt[1] * xdata) / np.sum(wt[1])
        # std 更新为 高斯分布的标准差。它是数据点偏离均值的加权平方和的平方根
        std = np.sqrt(np.sum(wt[1] * ((xdata - mu) ** 2) / sum(wt[1])))
        alpha, beta = update_gamma_pars(xdata, wt[0])
        params = [rate, alpha, beta, mu, std]
        new = dmix(xdata, params)
        loglik = np.sum(np.log10(new))
        eps = (loglik - loglik_old) ** 2
        loglik_old = loglik
        iter = iter + 1
        if iter > 100:
            break
    return params


def get_mix_parameters(count, point=np.log(1)):
    # point是用来算无效基因，以及当前基因表达为0的比例
    # count represents each type of cell, with rows representing cell changes and columns representing gene changes
    count = np.array(count)
    # The genes with very low expression were screened out
    genes_expr = abs(np.mean(count, axis=0) - point)  # 计算每个基因的均值，再减去point，然后取绝对值，得到一个一维的向量
    # 找到平均表达值低于 1e-2 的基因（即认为是无效基因）
    # 源 代码是0.01
    null_genes = np.argwhere(genes_expr < 1-2)  # 得到一个x行1列的array，x代表无效基因的数量

    # paralist represents 5 parameters of each gene
    paralist = np.zeros([count.shape[1], 5])  # 初始化一个矩阵，行是基因，列是 5 个参数："rate", "alpha", "beta", "mu", "sigma"
    for i in tqdm(range(count.shape[1])):
        if i in null_genes:  # use nan to fill in invalid genes
            paralist[i] = np.repeat(np.nan, 5)
        else:
            xdata = count[:, i]
            params = get_mix_internal(xdata, point)  # calculate the params of each columns
            paralist[i] = params
    #  输出每个基因的5个模型参数，和无效基因索引
    return paralist, null_genes


def dhat(xdata, params):
    rate, alpha, beta, mu, std = params
    gam = rate * gamma.pdf(xdata, a=alpha, scale=(1 / beta))
    nor = (1 - rate) * norm.pdf(xdata, mu, std)
    dropout = gam / (gam + nor)
    return dropout


def get_dropout_rate(count, point=np.log(1)):
    paralist, null_genes = get_mix_parameters(count, point)
    df = pd.DataFrame(paralist, columns=['rate', 'alpha', 'beta', 'mu', 'sigma'])
    # print(df)
    dropout_rate = np.zeros((count.shape[1], count.shape[0]))  # init the matrix
    print("Calculating dropout rate...")
    for i in tqdm(range(count.shape[1])):
        if not np.isnan(paralist[i][0]):
            dropout_rate[i] = dhat(count[:, i], paralist[i])
    null_genes = null_genes.flatten()
    return dropout_rate.T, null_genes


def cluster_get_dropout_rate(rna_ad: sc.AnnData, point=np.log(1.01)):
    rna_ad.X = np.log(1.01+rna_ad.X)

    import scipy.sparse as sp
    if sp.issparse(rna_ad.X):
        rna_ad.X = rna_ad.X.toarray()
    if cfg.IDENTIFY_USE_CELLTYPE:
        rna_ad.obs['cell_type'] = pd.Categorical(rna_ad.obs['cell_type']).codes
        cluster_ids = list(set(rna_ad.obs['cell_type'].tolist()))
        cluster_ids.sort()
        print('celltype_id:', cluster_ids)
        dropout_rate_list = []
        for c_id in cluster_ids:
            rna_ad_part = rna_ad[rna_ad.obs['cell_type'] == c_id, :].copy()
            # print(rna_ad_part.to_df())
            import scipy.sparse as sp
            dropout_rate_part = get_dropout_rate(rna_ad_part.X, point)[0]
            dropout_rate_part = pd.DataFrame(data=dropout_rate_part, index=rna_ad_part.obs_names.tolist(),
                                             columns=rna_ad_part.var_names.tolist())
            dropout_rate_part = sc.AnnData(dropout_rate_part)
            # print(dropout_rate_part.to_df())
            dropout_rate_list.append(dropout_rate_part)
    else:
        sc.pp.pca(rna_ad)
        sc.pp.neighbors(rna_ad)
        X_pca = rna_ad.obsm['X_pca']

        kmeans = KMeans(n_clusters=cfg.N_CLUSTER, random_state=0)
        print(cfg.N_CLUSTER)
        clusters = kmeans.fit_predict(X_pca)
        rna_ad.obs['kmeans'] = clusters.astype(str)
        cluster_ids = list(set(rna_ad.obs['kmeans'].tolist()))
        cluster_ids.sort()

        dropout_rate_list = []
        for c_id in cluster_ids:
            rna_ad_part = rna_ad[rna_ad.obs['kmeans'] == c_id, :].copy()
            # print(rna_ad_part.to_df())
            dropout_rate_part = get_dropout_rate(rna_ad_part.X, point)[0]
            dropout_rate_part = pd.DataFrame(data=dropout_rate_part, index=rna_ad_part.obs_names.tolist(), columns=rna_ad_part.var_names.tolist())
            dropout_rate_part = sc.AnnData(dropout_rate_part)
            # print(dropout_rate_part.to_df())
            dropout_rate_list.append(dropout_rate_part)
    dropout_rate = sc.concat(dropout_rate_list, axis=0)
    dropout_rate = dropout_rate[rna_ad.obs_names, rna_ad.var_names].copy()

    np.nan_to_num(dropout_rate.X, copy=False, nan=0)

    return dropout_rate.X



