import torch
import torch.autograd
import numpy as np
import utils


def image(img_pred, img_target, loss_weight=None):
    batch_size = img_pred.size(0)
    if loss_weight is None:
        image_loss = (((img_pred - img_target).abs()).sum()) / (256 * 256 * 3)
    else:
        image_loss = (((img_pred - img_target).abs()*loss_weight).abs().sum()) / (256 * 256 * 3)
    image_loss = image_loss*20/batch_size
    return image_loss


def feature(features, passthrough_dict, weight=None):
    passthrough = []
    for _, indices in passthrough_dict.items():
        passthrough += indices
    batch_size, num_feature = features["original_feature"].size()
    block = [i for i in range(num_feature) if i not in passthrough]
    feature_loss = (features["original_feature"] - features["selected_feature"])[:, block].abs().sum()/(batch_size * len(block))
    if weight:
        return feature_loss * weight
    return feature_loss


def change(features, indices_dict, passthrough_dict, weight=None):
    old_feature = features["original_feature"]
    change_feature_loss = {}
    for label, indices_list in indices_dict.items():
        change_feature_loss[label] = torch.autograd.Variable(torch.zeros(1).type(torch.FloatTensor)).cuda()
        for indices in indices_list:
            mini_feature = old_feature[indices, :]
            passthrough = passthrough_dict[label]
            mini_select = mini_feature[:, passthrough]
            change_feature_loss[label] += (mini_select - mini_select.mean(0).expand_as(mini_select).contiguous()).abs().sum()/(len(indices) * len(passthrough))
        change_feature_loss[label] /= len(indices_list)
    if weight:
        for label in change_feature_loss.keys():
            change_feature_loss[label] *= weight
    return change_feature_loss


def pixel(output, target, select=None):
    fun = torch.nn.MSELoss(size_average=True, reduce=True)
    batch_size = output.size(0)
    if select is None:
        pixel_loss = fun(output[0, :, :, :], target[0, :, :, :]) * 255 * 255
        for i in range(1, batch_size):
            pixel_loss += fun(output[i, :, :, :], target[i, :, :, :]) * 255 * 255
        return pixel_loss / batch_size
    else:
        select -= 1
        pixel_loss = fun(output[select, :, :, :], target[select, :, :, :]) * 255 * 255
        counter = 1
        for i in range(select, batch_size):
            if i % 5 != select:
                continue
            pixel_loss += fun(output[i, :, :, :], target[i, :, :, :]) * 255 * 255
            counter += 1
        return pixel_loss / counter
