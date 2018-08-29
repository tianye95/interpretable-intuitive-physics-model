from __future__ import print_function
import os
import shutil
import json
from subprocess import call

import numpy as np
import pickle
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_image_records(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def save_checkpoint(state, is_best, checkpoint_path, epoch):
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    epoch = epoch - epoch % 5
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint_' + str(int(epoch)) + '.pth')
    best_model_file = os.path.join(checkpoint_path, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def load_best_checkpoint(checkpoint_path, model, optimizer):
    best_model_file = os.path.join(checkpoint_path, 'model_best.pth')
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    if os.path.isfile(best_model_file):
        print("=> loading best model '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file, map_location=lambda storage, loc: storage.cuda())
        start_epoch = checkpoint['epoch']
        best_epoch_error = checkpoint['best_epoch_error']
        model.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            print("Fixed decoder")
            pass

        print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
        return start_epoch, best_epoch_error, model, optimizer
    else:
        print("=> no best model found at '{}'".format(best_model_file))
        return 0, float('Inf'), model, optimizer


def get_passthrough(trainlabels, parameter_length, vector_length):
    labels = ["mass", "force", "friction", "other"]
    passthroughDict = {}
    for label in trainlabels:
        parameter_id = labels.index(label)
        if label == "other":
            passthroughDict[label] = range(parameter_id * parameter_length, vector_length)
        else:
            passthroughDict[label] = range(parameter_id * parameter_length, (parameter_id + 1) * parameter_length)
    return passthroughDict


def combinebatchdata(batch, autograd, cuda):
    for i in batch.keys():
        batch[i] = torch.autograd.Variable(batch[i], requires_grad=autograd)
        if cuda:
            batch[i] = batch[i].cuda()
    return batch


def get_weight(batch, cuda):
    weight = {}
    for i in batch.keys():
        batch_mean = batch[i].clone().mean(1, keepdim=True).repeat(1, 3, 1, 1)
        image_diff = (batch[i] - batch_mean).abs()
        image_diff = image_diff.max(1, keepdim=True)[0].repeat(1, 3, 1, 1)
        weight[i] = image_diff
        if cuda:
            weight[i] = weight[i].cuda()
    return weight


def inverse_transform(img):
    img = np.transpose(img, (1, 2, 0)) + np.array([0.672, 0.668, 0.648])
    return np.clip(img, 0, 1)


def write_to_file(filename, info):
    if os.path.isfile(filename + ".json"):
        call(["rm", filename + ".json"])
    with open(filename + ".json", 'w') as json_file:
        json.dump(info, json_file)
        json_file.close()


def print_training_status(epoch, batch_id, batch_num, losses, otherlosses=None):
        print('Epoch: [{0}][{1}/{2}]||'
                'Loss {loss.val:.4f} ({loss.avg:.4f})||'
                .format(epoch, batch_id, batch_num, loss=losses),
                end=''
                )
        if otherlosses:
            for key, loss in otherlosses.items():
                if key == 'image':
                    continue
                print('{name:.4s} {loss.val:.4f} ({loss.avg:.4f})|'.format(name=key, loss=loss), end='')
        print('')

