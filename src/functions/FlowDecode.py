import torch.nn as nn
import torch.nn.functional as F
from LinearPermutate import LinearPermutate


class FlowDecode(nn.Module):
    def __init__(self, dim_feature=256):
        super(FlowDecode, self).__init__()
        self.fc = nn.Sequential(
            LinearPermutate(dim_feature, 1024),
            LinearPermutate(1024, 4096)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 256, 4, 2, 1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 2, 3, 1, 1),
            nn.Tanh()
        )
        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward(self, maskx, feature, label_and_indiceslist=None):
        num_batch = feature.size(0)
        output = self.fc(feature)
        output = output.view(num_batch, 64, 8, 8)
        grid = self.decoder(output).permute(0, 2, 3, 1)
        output = F.grid_sample(maskx, grid)
        return output, grid