import torch
import torch.nn
import torch.autograd


class LinearPermutate(torch.nn.Module):
    def __init__(self, input_channel, output_channel, args=None):
        super(LinearPermutate, self).__init__()
        self.args = args
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_channel, output_channel, bias=True),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        if len(x.size()) == 4:
            return self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            return self.linear(x)