import torch
from torch import nn

from trainer.config import cfg
from trainer.utils import init_weights


class Generator(nn.Module):
    def __init__(self, input_size=cfg.Z_SIZE, output_size=cfg.GENENUM):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, cfg.DE_SIZE[0]),  # 10->2000
            # nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(cfg.DE_SIZE[0], cfg.DE_SIZE[1]),  # 2000->500
            # nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(cfg.DE_SIZE[1], cfg.DE_SIZE[2]),  # 500->500
            # nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(cfg.DE_SIZE[2], output_size),  # 500->gene num
            # nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        init_weights(self)

    def forward(self, z_input):
        # label_onehot = torch.nn.functional.one_hot(labels, self.n_class)
        # label_onehot = label_onehot.float()
        # z_input = torch.cat((z_input, label_onehot), dim=1)
        z_input = self.model(z_input)

        return z_input


if __name__ == '__main__':
    generator = Generator(10, 969)
    input_image = torch.randn(2, 10)
    output = generator(input_image)
    print(output.shape)
