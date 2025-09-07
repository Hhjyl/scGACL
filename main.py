import copy
import time
import torch.autograd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import scanpy as sc

from evaluation.dropout_identify import cluster_get_dropout_rate
from trainer.config import cfg
from trainer.datasets import MyDataset
from trainer.train import train_model, reconstruct_data, train_VAE, pretrain_VAE
from trainer.utils import mkdir_p

if __name__ == '__main__':
    start_time_total = time.time()
    # torch.autograd.set_detect_anomaly(True)  # 会减慢运行速度
    device = cfg.DEVICE

    # 评估模型的数据
    drop_preprocess_path = r'F:/pythonProject/scGACL/dataset/sim.groups1_preprocess.h5ad'
    drop_raw_path = r'F:/pythonProject/scGACL/dataset/sim.groups1_counts.csv'

    # 读取数据，构建数据集
    rna_ad = sc.read_h5ad(drop_preprocess_path)
    print(rna_ad.to_df())
    dataset = MyDataset(ad=rna_ad, use_label=cfg.MODEL_USE_CELLTYPE)
    train_dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.PRE_BATCH_SIZE, shuffle=False)

    # 
    out_path = r'F:/pythonProject/scGACL/sim1/'
    mkdir_p(out_path)
    net_path = out_path + 'net_state_dict/'
    mkdir_p(net_path)
    pretrain_VAE(train_loader=train_dataloader, pre_train_path=net_path, log_path=out_path, device=device)

    # 训练VAE模型
    train_dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.VAE_BATCH_SIZE, shuffle=False)
    train_VAE(train_loader=train_dataloader, net_path=net_path, log_path=out_path, device=device, rna_ad=rna_ad, )

    # 联合训练模型
    train_dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
    train_model(train_loader=train_dataloader, net_path=net_path, log_path=out_path, device=device, rna_ad=rna_ad, )
    res_df = reconstruct_data(train_dataloader, net_path, rna_ad, device, None, None)

    # 识别dropout点
    drop_raw = sc.read(drop_raw_path).T
    drop_raw = drop_raw[rna_ad.obs_names.tolist(), rna_ad.var_names.tolist()].copy()
    drop_raw.obs = rna_ad.obs
    predict_drop = cluster_get_dropout_rate(copy.deepcopy(drop_raw), np.log(1.01))
    threshold = cfg.DROP_THRESHOLD

    genes = rna_ad.var_names.tolist()
    cells = rna_ad.obs_names.tolist()

    rna_predict = sc.AnnData(res_df).X
    rna_predict = np.where(drop_raw.X == 0, rna_predict, rna_ad.X)

    rna_predict = np.where((predict_drop < threshold) & (drop_raw.X == 0), 0, rna_predict)
    rna_impute = pd.DataFrame(rna_predict, index=cells, columns=genes)
    rna_impute.to_csv(out_path + "result.csv")

