import torch
from torch import nn

from trainer.config import cfg
from trainer.utils import init_weights


class Discriminator(nn.Module):
    def __init__(self, input_size=cfg.GENENUM):
        super(Discriminator, self).__init__()

        # __C.DI_SIZE = [500, 500, 2000]
        self.model = nn.Sequential(
            nn.Linear(input_size, cfg.DI_SIZE[0]),
            nn.BatchNorm1d(cfg.DI_SIZE[0]),
            nn.LeakyReLU(),

            nn.Linear(cfg.DI_SIZE[0], cfg.DI_SIZE[1]),
            nn.BatchNorm1d(cfg.DI_SIZE[1]),
            nn.LeakyReLU(),

            nn.Linear(cfg.DI_SIZE[1], cfg.DI_SIZE[2]),
            nn.BatchNorm1d(cfg.DI_SIZE[2]),
            nn.LeakyReLU(),

        )
        self.dis = nn.Sequential(
            nn.Linear(cfg.DI_SIZE[2], 1),
            nn.Sigmoid())
        self.proj1 = nn.Sequential(
            nn.Linear(cfg.DI_SIZE[2], cfg.PROJECT_SIZE1)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(cfg.DI_SIZE[2], cfg.PROJECT_SIZE2)
        )
        init_weights(self)

    def forward(self, z_input):
        output = self.model(z_input)
        adv = self.dis(output)
        proj1 = self.proj1(output)
        proj2 = self.proj2(output)
        return adv, proj1, proj2



