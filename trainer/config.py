import torch
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.Z_SIZE = 10  # 10
__C.PROJECT_SIZE1 = 10  # 10
__C.PROJECT_SIZE2 = 10  # 10
__C.TEMPERATURE = 0.9  # 0.9
__C.CUDA = True
__C.IDENTIFY_USE_CELLTYPE = True
__C.MODEL_USE_CELLTYPE = True
__C.GENENUM = 969
__C.N_CLASS = 4
__C.N_CLUSTER = 4
__C.DEVICE = torch.device("cuda:2")
__C.DROP_THRESHOLD = 0.5

__C.EN_SIZE = [500, 500, 2000]  # 512，256
__C.DE_SIZE = [2000, 500, 500]  # 256，512
__C.DI_SIZE = [500, 500, 2000]  # 512，256

__C.TRAIN = edict()
__C.TRAIN.B1 = 0.5
__C.TRAIN.B2 = 0.99
__C.TRAIN.DECAY = 2.5 * 1e-5

__C.TRAIN.PRE = True
__C.TRAIN.PRE_EPOCH = 100  # 100.
__C.TRAIN.PRE_LR = 1 * 10 ** (-3)
__C.TRAIN.PRE_BATCH_SIZE = 32

__C.TRAIN.VAEEPOCH = 100  # 100
__C.TRAIN.VAELR = 1 * 10 ** (-3)  # It cannot be too large
__C.TRAIN.VAE_BATCH_SIZE = 32


__C.TRAIN.EPOCH = 100  # 100
__C.TRAIN.LR = 1 * 10 ** (-4)  # It cannot be too large
__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.LAMBDA_G_ADV = 1
__C.TRAIN.LAMBDA_SIMCL = 1
__C.TRAIN.LAMBDA_SUPCL = 1  # 1
