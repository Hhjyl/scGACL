import torch
from torch import nn

from trainer.config import cfg
from trainer.utils import init_weights


class Encoder(nn.Module):
    def __init__(self, input_size=cfg.GENENUM, output_size=cfg.Z_SIZE,):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, cfg.EN_SIZE[0]),
            nn.BatchNorm1d(cfg.EN_SIZE[0]),
            nn.ReLU(),

            nn.Linear(cfg.EN_SIZE[0], cfg.EN_SIZE[1]),
            nn.BatchNorm1d(cfg.EN_SIZE[1]),
            nn.ReLU(),

            nn.Linear(cfg.EN_SIZE[1], cfg.EN_SIZE[2]),
            nn.BatchNorm1d(cfg.EN_SIZE[2]),
            nn.ReLU(),
        )
        self.mu = nn.Linear(cfg.EN_SIZE[2], output_size)
        self.logvar = nn.Linear(cfg.EN_SIZE[2], output_size)
        init_weights(self)

    def forward(self, z_input):
        # z_input = torch.cat((z_input, label_onehot), dim=1)

        z_input = self.model(z_input)
        mu = self.mu(z_input)
        logvar = self.logvar(z_input)

        # if torch.isnan(mu).any():
        #     mu = torch.nan_to_num(mu, nan=0.0)
        # if torch.isnan(logvar).any():
        #     logvar = torch.nan_to_num(logvar, nan=0.0)

        #
        sigma = torch.exp(logvar * 0.5)
        std_z = torch.randn_like(mu)
        z = mu + (sigma * std_z)

        return z, mu, logvar


if __name__ == '__main__':
    encoder = Encoder()
    input_image = torch.randn(2, 969)
    output = encoder(input_image)
    print(output[0].shape)
