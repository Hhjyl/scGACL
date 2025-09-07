from __future__ import print_function

import os
import time
from itertools import chain
import numpy as np
import pandas as pd
import torch
import numpy
import random

from sklearn.mixture import GaussianMixture

from models.Encoder import Encoder
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.GMM import GMM
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR
from trainer.config import cfg
from trainer.utils import MSEloss, cluster_acc, mkdir_p, GMMLoss, G_ADV_loss, D_ADV_loss, CL_loss

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)


def pretrain_VAE(train_loader, pre_train_path, log_path, device):
    z_size = cfg.Z_SIZE  # 隐变量的维度
    n_gene = cfg.GENENUM
    pre_epochs = cfg.TRAIN.PRE_EPOCH
    pre_lr = cfg.TRAIN.PRE_LR  # 学习率
    n_cluster = cfg.N_CLUSTER
    decay = cfg.TRAIN.DECAY
    b1, b2 = cfg.TRAIN.B1, cfg.TRAIN.B2

    # -------------初始化模型，并移动到GPU上---------#
    netE = Encoder(input_size=n_gene, output_size=z_size).to(device)
    netG = Generator(input_size=z_size, output_size=n_gene).to(device)
    gmm = GMM(n_cluster=n_cluster, hid_dim=z_size).to(device)

    # ------------ optimizers -------------#
    optimizer = Adam(chain(netG.parameters(), netE.parameters(), ), lr=pre_lr, weight_decay=decay, betas=(b1, b2))

    # ------------ pretrain models -------------#
    loss_list = []
    for epoch in range(pre_epochs):
        L = 0
        start_t = time.time()
        for index, (x, y) in enumerate(train_loader):
            x = x.to(device)
            _, z, _ = netE(x)
            x_ = netG(z)
            loss = MSEloss(x, x_, device)
            L += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_t = time.time()
        loss_list.append(L)

    netE.logvar.load_state_dict(netE.mu.state_dict())

    _gmm = GaussianMixture(n_components=n_cluster, covariance_type='diag', reg_covar=1e-5)
    Z = []
    with torch.no_grad():
        for index, (x, y) in enumerate(train_loader):
            x = x.to(device)

            _, z, _ = netE(x)
            Z.append(z)
    Z = torch.cat(Z, 0).detach().cpu().numpy()
    pre = _gmm.fit_predict(Z)

    gmm.pi_.data = torch.from_numpy(_gmm.weights_).to(device).float()
    gmm.mu_c.data = torch.from_numpy(_gmm.means_).to(device).float()
    gmm.log_sigma2_c.data = torch.log(torch.from_numpy(_gmm.covariances_).to(device).float())

    mkdir_p(pre_train_path)
    torch.save(netE.state_dict(), os.path.join(pre_train_path, 'pretrain_netE.pth'))
    torch.save(netG.state_dict(), os.path.join(pre_train_path, 'pretrain_netG.pth'))
    torch.save(gmm.state_dict(), os.path.join(pre_train_path, 'pretrain_gmm.pth'))


def train_VAE(train_loader, net_path, log_path, device, rna_ad, data_path=None):
    print('train VAE')
    z_size = cfg.Z_SIZE  # 隐变量的维度
    n_gene = cfg.GENENUM
    n_epochs = cfg.TRAIN.VAEEPOCH
    lr = cfg.TRAIN.VAELR  # 学习率
    n_cluster = cfg.N_CLUSTER
    decay = cfg.TRAIN.DECAY
    b1, b2 = cfg.TRAIN.B1, cfg.TRAIN.B2

    # -------------初始化模型，并移动到GPU上---------#
    netE = Encoder(input_size=n_gene, output_size=z_size).to(device)
    netG = Generator(input_size=z_size, output_size=n_gene).to(device)
    gmm = GMM(n_cluster=n_cluster, hid_dim=z_size).to(device)
    if cfg.TRAIN.PRE:
        netE.load_state_dict(torch.load(os.path.join(net_path, "pretrain_netE.pth")))
        netG.load_state_dict(torch.load(os.path.join(net_path, "pretrain_netG.pth")))
        gmm.load_state_dict(torch.load(os.path.join(net_path, "pretrain_gmm.pth")))

    # ------------ optimizers -------------#
    optimizer_encoder = Adam(params=netE.parameters(), lr=lr, weight_decay=decay, betas=(b1, b2))
    optimizer_gmm = Adam(params=gmm.parameters(), lr=lr, weight_decay=decay, betas=(b1, b2))
    optimizer_generator = Adam(params=netG.parameters(), lr=lr, weight_decay=decay, betas=(b1, b2))

    # ------------ train model -------------#

    for i in range(n_epochs):
        start_t = time.time()

        MSE_loss_epoch = 0
        GMM_loss_epoch = 0
        ELBO_loss_epoch = 0

        for j, (x, y) in enumerate(train_loader):
            netE.train()
            netG.train()
            gmm.train()
            x = x.float().to(device)
            y = y.float().to(device)

            # 编码器、生成器、高斯混合模型的损失
            z, z_mu, z_sigma2_log = netE(x)
            x_tilde = netG(z)
            l_mse = MSEloss(x, x_tilde, device)
            l_gmm = GMMLoss(z_mu, z_sigma2_log, gmm)
            l_elbo = l_mse + l_gmm
            ELBO_loss_epoch = l_elbo.item() + ELBO_loss_epoch
            MSE_loss_epoch = MSE_loss_epoch + l_mse.item()
            GMM_loss_epoch = GMM_loss_epoch + l_gmm.item()
            netE.zero_grad()
            netG.zero_grad()
            gmm.zero_grad()
            l_elbo.backward()
            optimizer_encoder.step()
            optimizer_generator.step()
            optimizer_gmm.step()

        # 评价模型性能，先eval，然后重构数据，再调用函数算指标
        netE.eval()
        netG.eval()
        # with torch.no_grad():
        #     predict_df = reconstruct_data(test_loader=train_loader, net_path=None, rna_ad=rna_ad, device=device,
        #                                   netE=netE, netG=netG)
        #     evaluate_cluster(predict_df, rna_ad, impute=True)

        #     rmse, cosin_sim, pcc = evaluate_real_mask(truecount_path=data_path['truecount'],
        #                                            drop_place_path=data_path['dropplace'], rna_predict=predict_df,)
        #

    torch.save(netE.state_dict(), net_path + r'trainVAE_netE.pth')
    torch.save(netG.state_dict(), net_path + r'trainVAE_netG.pth')
    torch.save(gmm.state_dict(), net_path + r'trainVAE_gmm.pth')


def train_model(train_loader, net_path, log_path, device, rna_ad, data_path=None):
    print('train')
    z_size = cfg.Z_SIZE  # 隐变量的维度
    n_gene = cfg.GENENUM
    n_epochs = cfg.TRAIN.EPOCH
    lr = cfg.TRAIN.LR  # 学习率
    n_cluster = cfg.N_CLUSTER
    decay = cfg.TRAIN.DECAY
    b1, b2 = cfg.TRAIN.B1, cfg.TRAIN.B2

    # -------------初始化模型，并移动到GPU上---------#
    netE = Encoder(input_size=n_gene, output_size=z_size).to(device)
    netE.load_state_dict(torch.load(os.path.join(net_path, "trainVAE_netE.pth")))
    netG = Generator(input_size=z_size, output_size=n_gene).to(device)
    netG.load_state_dict(torch.load(os.path.join(net_path, "trainVAE_netG.pth")))
    gmm = GMM(n_cluster=n_cluster, hid_dim=z_size).to(device)
    gmm.load_state_dict(torch.load(os.path.join(net_path, "trainVAE_gmm.pth")))
    netD = Discriminator(input_size=n_gene).to(device)
    # netD.load_state_dict((torch.load(os.path.join(net_path, "netD.pth"))))

    # ------------ optimizers -------------#
    optimizer_encoder = Adam(params=netE.parameters(), lr=lr, weight_decay=decay, betas=(b1, b2))
    optimizer_gmm = Adam(params=gmm.parameters(), lr=lr, weight_decay=decay, betas=(b1, b2))
    optimizer_generator = Adam(params=netG.parameters(), lr=lr, weight_decay=decay, betas=(b1, b2))
    optimizer_discriminator = Adam(params=netD.parameters(), lr=lr, weight_decay=decay, betas=(b1, b2))

    lr_en = StepLR(optimizer_encoder, step_size=50, gamma=0.95)
    lr_gen = StepLR(optimizer_generator, step_size=50, gamma=0.95)
    lr_gmm = StepLR(optimizer_gmm, step_size=50, gamma=0.95)

    # ------------ train model -------------#
    p_gen = 0.1
    for i in range(n_epochs):
        start_t = time.time()

        Discriminator_loss_epoch = 0
        D_ADV_loss_epoch = 0
        MSE_loss_epoch = 0
        G_ADV_loss_epoch = 0
        SIMCL_loss_epoch = 0
        SUPCL_loss_epoch = 0
        CL_loss_epoch = 0
        Generator_loss_epoch = 0
        GMM_loss_epoch = 0
        ELBO_loss_epoch = 0

        for j, (x, y) in enumerate(train_loader):
            netE.train()
            netG.train()
            gmm.train()
            netD.train()

            x = x.float().to(device)
            y = y.float().to(device)

            # 编码器、生成器、高斯混合模型的损失
            z, z_mu, z_sigma2_log = netE(x)
            x_tilde = netG(z)
            l_mse = MSEloss(x, x_tilde, device)
            l_gmm = GMMLoss(z_mu, z_sigma2_log, gmm)
            l_elbo = l_mse + l_gmm
            ELBO_loss_epoch = l_elbo.item() + ELBO_loss_epoch
            MSE_loss_epoch = MSE_loss_epoch + l_mse.item()
            GMM_loss_epoch = GMM_loss_epoch + l_gmm.item()
            netE.zero_grad()
            netG.zero_grad()
            gmm.zero_grad()
            l_elbo.backward()
            optimizer_encoder.step()
            optimizer_generator.step()
            optimizer_gmm.step()

            # 生成器对抗损失
            l_g_adv = (G_ADV_loss(x, netE, netG, netD, device)) * cfg.TRAIN.LAMBDA_G_ADV
            netG.zero_grad()
            netE.zero_grad()
            gmm.zero_grad()
            l_g_adv.backward()
            optimizer_encoder.step()
            optimizer_generator.step()
            optimizer_gmm.step()
            G_ADV_loss_epoch = G_ADV_loss_epoch + l_g_adv.item()

            # 判别器的对抗损失
            l_d_adv = D_ADV_loss(x, y, netE, netG, netD, device) * cfg.TRAIN.LAMBDA_G_ADV
            D_ADV_loss_epoch += l_d_adv.item()
            netD.zero_grad()
            l_d_adv.backward()
            optimizer_discriminator.step()

            # 对比损失
            simclr, supcon = CL_loss(x, y, netE, netG, netD, device, p_gen)
            l_cl = simclr * cfg.TRAIN.LAMBDA_SIMCL + supcon * cfg.TRAIN.LAMBDA_SUPCL
            netG.zero_grad()
            netE.zero_grad()
            netD.zero_grad()
            gmm.zero_grad()
            l_cl.backward()
            optimizer_encoder.step()
            optimizer_generator.step()
            optimizer_gmm.step()
            optimizer_discriminator.step()
            SIMCL_loss_epoch += simclr.item()
            SUPCL_loss_epoch += supcon.item()
            CL_loss_epoch += l_cl.item()

        if i % 10 == 0:
            p_gen = min(1.0, p_gen + (1.0 - 0.1) / (n_epochs / 10))  # 以前是500/10

        # lr_en.step()
        # lr_gen.step()
        # lr_gmm.step()

        Generator_loss_epoch = ELBO_loss_epoch + G_ADV_loss_epoch + CL_loss_epoch

        # 评价模型性能，先eval，然后重构数据，再调用函数算指标
        netE.eval()
        netG.eval()

        # 每30个epoch保存模型
        if (i + 1) % 10 == 0:
            check_path = os.path.join(net_path, "checkpoint_{}//".format(i + 1))
            mkdir_p(check_path)
            torch.save(netE.state_dict(), os.path.join(check_path, 'netE.pth'))
            torch.save(netG.state_dict(), os.path.join(check_path, 'netG.pth'))
            torch.save(gmm.state_dict(), os.path.join(check_path, 'gmm.pth'))
            torch.save(netD.state_dict(), os.path.join(check_path, 'netD1.pth'))

    torch.save(netE.state_dict(), net_path + r'netE.pth')
    torch.save(netG.state_dict(), net_path + r'netG.pth')
    torch.save(gmm.state_dict(), net_path + r'gmm.pth')
    torch.save(netD.state_dict(), net_path + r'netD.pth')


def reconstruct_data(test_loader, net_path, rna_ad, device, netE, netG):
    if net_path:
        netE = Encoder(input_size=cfg.GENENUM, output_size=cfg.Z_SIZE).to(device)
        state_dictE = torch.load(net_path + r'netE.pth')
        netE.load_state_dict(state_dictE)
        netG = Generator(input_size=cfg.Z_SIZE, output_size=cfg.GENENUM).to(device)
        state_dictG = torch.load(net_path + r'netG.pth')
        netG.load_state_dict(state_dictG)

    impute_result = []
    for j, (x, label) in enumerate(test_loader):
        netE.eval()
        netG.eval()
        x = x.float().to(device)
        x.requires_grad_(False)

        z, _, _ = netE(x)
        x_rec = netG(z)

        for i in range(x_rec.shape[0]):
            im = x_rec[i].data.cpu().numpy()
            im = im.reshape((1, x_rec.shape[1]))
            impute_result.append(im)

    result_ar = np.vstack(impute_result)
    res_df = pd.DataFrame(data=result_ar, index=rna_ad.obs_names.tolist(),
                          columns=rna_ad.var_names.tolist())

    return res_df
