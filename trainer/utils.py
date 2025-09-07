import errno
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer.config import cfg
import scanpy as sc
from trainer.datasets import MyDataset
import torch.nn.functional as F
from trainer.SCL_loss import SupConLoss


def mkdir_p(path):
    try:
        os.makedirs(path)  # 创建多层目录（单层用os.mkdir)
    except OSError as exc:  # Python >2.5  捕获 OSError 异常
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass  # 如果目录已存在，且路径是目录，则忽略异常。
        else:
            raise  # 否则，重新抛出异常。


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def MSEloss(x, x_tilde, device):
    mse_loss = nn.MSELoss()

    # 创建掩码，假设真实样本中非零位置由掩码指示
    mask = (x != 0).float()  # 掩码，其中非零位置为1，零位置为0
    mask = mask.to(device)
    x_tilde_m = x_tilde * mask
    # 计算真实样本和生成样本之间的MSE，仅考虑非零位置
    reconstruction_loss = mse_loss(x, x_tilde_m)
    return reconstruction_loss


def nt_xent(out1, out2, temperature=cfg.TEMPERATURE, normalize=True):
    if normalize:
        out1 = torch.nn.functional.normalize(out1)
        out2 = torch.nn.functional.normalize(out2)
    N = out1.size(0)

    _out = [out1, out2]
    # 一批真实数据对应的2批增强投影数据拼接在一起
    outputs = torch.cat(_out, dim=0)

    # 因为数据已经归一化，所以此时矩阵相乘结果就是每个样本和其他所有样本的余弦相似度
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    # 将一个样本和自身算的相似度置为很大的负数
    sim_matrix.fill_diagonal_(-5e4)
    # 所有元素取指数相除 再取对数
    sim_matrix = torch.nn.functional.log_softmax(sim_matrix, dim=1)
    # 加号前是第一批增强数据对应第二批增强数据的正样本对，加号后是第二批增强数据对应第一批增强数据的正样本对
    loss = -torch.sum(sim_matrix[:N, N:].diag() + sim_matrix[N:, :N].diag()) / (2 * N)

    return loss


def sample_generated_samples(generated_samples, labels, p_gen):
    """
    从生成样本中随机抽取部分样本
    :param generated_samples: torch tensor，形状为 [N, ...]
    :param p_gen: 抽样比例，范围 [0, 1]
    :return: 抽取后的生成样本
    """
    num_samples = generated_samples.size(0)
    num_select = int(p_gen * num_samples)
    # 使用 torch.randperm 生成随机索引，并选择前 num_select 个
    indices = torch.randperm(num_samples)[:num_select]
    return generated_samples[indices], labels[indices]


def CL_loss(x, y, netE, netG, netD, device, p_gen):
    batch_size = len(x)
    real_logits, real_proj1, real_proj2 = netD(x)

    z, mus, variances = netE(x)
    x_tilde = netG(z)
    fake_logits, fake_proj1, fake_proj2 = netD(x_tilde)

    simclr_loss = nt_xent(real_proj1, fake_proj1)

    fake_proj2_part, labels_part = sample_generated_samples(fake_proj2, y, p_gen)
    supcon = SupConLoss()
    supcon = supcon.to(device)
    proj2 = torch.cat([real_proj2, fake_proj2_part], dim=0)
    labels = torch.cat([y, labels_part])

    supcon_loss = supcon(proj2, labels)

    return simclr_loss, supcon_loss


def D_ADV_loss(x, y, netE, netG, netD, device):
    criterion = nn.BCELoss()
    batch_size = len(x)
    real_labels = 0.9 + 0.1 * torch.rand(batch_size)
    real_labels.to(device)
    fake_labels = 0.1 * torch.rand(batch_size)
    fake_labels.to(device)
    real_labels = real_labels.unsqueeze(1).to(device)
    fake_labels = fake_labels.unsqueeze(1).to(device)

    real_logits, _, _ = netD(x)
    errD_real = criterion(real_logits, real_labels)

    z, mus, variances = netE(x)
    x_tilde = netG(z)
    # print(x_tilde)
    fake_logits, _, _ = netD(x_tilde)
    errD_fake = criterion(fake_logits, fake_labels)
    errD = errD_real + errD_fake

    return errD


def G_ADV_loss(x, netE, netG, netD, device):
    z, z_mu, z_sigma2_log = netE(x)
    x_tilde = netG(z)

    fake_logits, _, _ = netD(x_tilde)
    batch_size = x.shape[0]
    criterion = nn.BCELoss()
    real_labels = 0.9 + 0.1 * torch.rand(batch_size)
    real_labels.to(device)
    real_labels = real_labels.unsqueeze(1).to(device)
    errD_fake = criterion(fake_logits, real_labels)

    return errD_fake


def GMMLoss(z_mu, z_sigma2_log, gmm):
    z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
    pi = gmm.pi_
    mu_c = gmm.mu_c
    log_sigma2_c = gmm.log_sigma2_c

    # 加小常数增加计算稳定性
    det = 1e-10
    # epsilon = 1e-6  # 小正数
    # epsilon = torch.tensor(epsilon, device=cfg.DEVICE, dtype=torch.float)
    pi = pi.clamp(min=1e-8)
    pi = pi / pi.sum()  # 重新归一化
    # log_sigma2_c = torch.clamp(log_sigma2_c, min=epsilon)
    # log_yita_c = torch.log(pi.unsqueeze(0)) + gmm.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
    # log_yita_c = log_yita_c - torch.logsumexp(log_yita_c, dim=1, keepdim=True)  # 归一化到后验概率
    # yita_c = torch.exp(log_yita_c)  # 转回概率空间

    # 计算每个数据点属于每个聚类的后验概率 yita_c
    # yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + gmm.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det
    # yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters
    logpi=torch.log(pi.unsqueeze(0))
    gpl=gmm.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
    log_yita_c = logpi + gpl
    if torch.isnan(logpi).any():
        print(logpi)
        print(pi)
    # assert not torch.isnan(gpl).any()
    # assert not torch.isnan(logpi).any()
    # assert not torch.isnan(log_yita_c).any()
    yita_c = torch.exp(log_yita_c) + det  # + det 转移到稳定区间
    if torch.isnan(yita_c).any():
        print(yita_c)
    yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # 归一化

    Loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                         torch.exp(z_sigma2_log.unsqueeze(1) -
                                                                   log_sigma2_c.unsqueeze(0)) +
                                                         (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) /
                                                         torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

    log_pi = torch.log(pi.unsqueeze(0))
    log_yita_c = torch.log(yita_c)
    # print(log_yita_c)
    # assert not torch.isnan(z_sigma2_log).any()
    # assert not torch.isnan(pi).any()
    # assert not torch.isnan(yita_c).any()
    # assert not torch.isnan(log_yita_c).any()
    # assert not torch.isnan(log_pi).any()

    Loss -= torch.mean(torch.sum(yita_c * (log_pi - log_yita_c), 1)) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
    return Loss


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w



