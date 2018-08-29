import torch
import torch.nn
import torch.autograd
import numpy as np
from torch.autograd.function import Function


class ProgressiveSelect(Function):
    def __init__(self, active, passthrough, indices_dict, weight=True):
        super(ProgressiveSelect, self).__init__()
        self.active = active
        self.passthrough = passthrough
        self.indices_dict = indices_dict
        self.weight = weight

    def forward(self, input):
        self.save_for_backward(input)
        if self.active:
            output = input.clone()
            other_passthrough = [p for l, p in self.passthrough.items() if l not in self.indices_dict.keys()]
            for label, indices_list in self.indices_dict.items():
                passthrough = self.passthrough[label]
                for indices in indices_list:
                    data = output[indices, :].clone()
                    mini_output = data.mean(0).expand_as(data).contiguous()
                    mini_output[:, passthrough] = data[:, passthrough]
                    for p in other_passthrough:
                        mini_output[:, p] = torch.zeros(mini_output[:, p].size())
                    output[indices, :] = mini_output
            self.mean = output
            return output
        else:
            return input

    def backward(self, grad_outputs):
        input, = self.saved_tensors
        if self.active:
            grad_input = (input.clone() - self.mean)
            other_passthrough = [p for l, p in self.passthrough.items() if l not in self.indices_dict.keys()]
            for label, indices_list in self.indices_dict.items():
                if label in self.passthrough:
                    indices = sum(indices_list, [])
                    passthrough = self.passthrough[label]
                    block = [i for i in range(grad_input.size(1)) if i not in passthrough]
                    mini_grad_output = grad_outputs[indices, :].clone()
                    mini_grad_input = grad_input[indices, :].clone()

                    mini_grad_select = mini_grad_output[:, passthrough].clone()
                    mini_grad_block = mini_grad_input[:, block].clone()

                    if self.weight:
                        select_range_mid = np.max(np.log10(np.abs((mini_grad_select.cpu().numpy()))))
                        mean_range_mid = np.max(np.log10(np.abs((mini_grad_block.cpu().numpy()))))
                        weight_pow = np.floor(select_range_mid - mean_range_mid) - 2
                        mini_grad_input *= 10 ** weight_pow

                    mini_grad_input[:, passthrough] = mini_grad_output[:, passthrough]
                    for p in other_passthrough:
                        mini_grad_input[:, p] = torch.zeros(mini_grad_input[:, p].size())
                    grad_input[indices, :] = mini_grad_input
        else:
            grad_input = grad_outputs.clone()
        return grad_input


class SelectiveFilter(torch.nn.Module):
    def __init__(self, active, passthrough):
        super(SelectiveFilter, self).__init__()
        self.active = active
        self.passthrough = passthrough

    def forward(self, input, indices_dict):
        fun = ProgressiveSelect(self.active, self.passthrough, indices_dict)
        return fun(input)

