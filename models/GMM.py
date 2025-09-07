import numpy as np
import torch
from torch import nn
from trainer.config import cfg


class GMM(nn.Module):

    def __init__(self, n_cluster=cfg.N_CLUSTER, hid_dim=cfg.Z_SIZE):
        super(GMM, self).__init__()

        self.n_cluster = n_cluster
        # self.r = r
        self.hid_dim = hid_dim

        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_cluster, ).fill_(1) / self.n_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.n_cluster, self.hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.n_cluster,
                                                           self.hid_dim).fill_(0), requires_grad=True)

    def predict(self, z):
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        # 计算每个数据点属于每个聚类的后验概率。 yita_c是一个矩阵，其行数等于输入数据x的样本数，列数等于聚类数nClusters。 yita_c[i,j]表示第i个数据点属于第j个聚类的概率.
        # self.gaussian_pdfs_log(z, mu_c, log_sigma2_c):计算每个数据点在每个高斯分布下的对数概率密度。
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))

        yita = yita_c.detach().cpu().numpy()
        # 将张量yita_c转换为NumPy数组yita，并使用np.argmax函数找到每一行中概率最大的列索引，即预测的聚类标签
        return np.argmax(yita, axis=1)

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        assert not torch.isnan(log_sigma2s).any()
        assert not torch.isnan(mus).any()
        assert not torch.isnan(x).any()#有nan，调小学习率、批次大小，supcon权重设置为1
        # 计算输入数据 x 在高斯混合模型所有分量上的对数概率密度
        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c+1, :], log_sigma2s[c:c+1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        # 计算输入数据 x 在单个多维高斯分布下的对数概率密度函数
        """
        输入:
        x: 形状为 (N, D) 的张量，表示 N 个 D 维的数据点。
        mu: 形状为 (1, D) 的张量，表示高斯分布的均值。
        log_sigma2: 形状为 (1, D) 的张量，表示高斯分布的对数方差。
        """
        return -0.5*(torch.sum(torch.tensor(np.log(np.pi*2), dtype=torch.float).to(cfg.DEVICE) +
                               log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2), 1))

