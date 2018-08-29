from __future__ import print_function, division

import os
import pickle
import warnings
import cv2
import scipy.misc
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
warnings.filterwarnings("ignore")
import random
random.seed(1)


class MODE:
    TRAIN, VALIDATION, TEST = 0, 1, 2


class Datasetmix(Dataset):
    def __init__(self, data_path, info_path, labels, mode, inframe, outframe,
                 input_size, output_size, initial=(100, 150), readsize=(224, 360), batch_sizes={}):

        # data settings
        self.data, self.batch_list, self.label_indices = [], [], {}
        self.data_path = data_path
        self.data_pickle = os.path.join(info_path, 'data.p')
        self.batch_pickle = os.path.join(info_path, 'batch.p')
        self.total_batch_size, self.num_batch = 0, float('Inf')

        # image settings
        self.input_size, self.output_size = input_size, output_size
        self.inframe, self.outframe = inframe, outframe
        self.h0, self.h1, self.w0, self.w1 = initial[0], initial[0] + readsize[0], initial[1], initial[1] + readsize[1]

        # transform settings
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(np.array([0.672, 0.668, 0.648]), np.array([1, 1, 1]))
        ])

        # read data information
        with open(self.data_pickle, 'r') as data_file:
            self.data = pickle.load(data_file)
            data_file.close()

        # read batch information
        with open(self.batch_pickle, 'r') as batch_file:
            self.batch_info = pickle.load(batch_file)
            for label in labels:
                mini_batch = self.batch_info[label][mode]
                if len(mini_batch) <= 0:
                    continue
                num_data_mini_batch = len(mini_batch[0])
                batch_size = batch_sizes[label] if label in batch_sizes else 1
                random.shuffle(mini_batch)
                batch = [sum([item for item in mini_batch[i:i+batch_size]], [])
                         for i in range(0, len(mini_batch)-batch_size, batch_size)]
                random.shuffle(batch)
                self.batch_list.append(batch)
                self.num_batch = min(len(batch), self.num_batch)
                self.label_indices[label] = [range(self.total_batch_size + i * num_data_mini_batch,
                                                  self.total_batch_size + (i + 1) * num_data_mini_batch)
                                            for i in range(batch_size)]
                self.total_batch_size += batch_size*num_data_mini_batch
            self.batch = [sum(batch, []) for batch in zip(*self.batch_list)]

            batch_file.close()

    def __len__(self):
        return self.num_batch

    def __getitem__(self, id):
        inimginfo = {}
        outimginfo = {}

        for f in self.inframe:
            inimginfo[f] = torch.zeros(self.total_batch_size, 3, self.input_size[1], self.input_size[0])
        for f in self.outframe:
            outimginfo[f] = torch.zeros(self.total_batch_size, 3, self.output_size[1], self.output_size[0])

        batch_info = self.batch[id]
        for iinfo, data_id in enumerate(batch_info):
            img_path = os.path.join(self.data_path, self.data[data_id][0])
            for f in self.inframe:
                img_name = os.path.join(img_path, '{:02d}.png'.format(f))
                image = self.read_image(img_name, self.input_size)
                inimginfo[f][iinfo, :, :, :] = self.transform(image)

            for f in self.outframe:
                img_name = os.path.join(img_path, '{:02d}.png'.format(f))
                image = self.read_image(img_name, self.output_size)
                outimginfo[f][iinfo, :, :, :] = self.transform(image)

        return inimginfo, outimginfo, self.label_indices

    def read_image(self, img_name, img_size=(224, 224)):
        image = scipy.misc.imread(img_name, mode='RGB')
        image = cv2.resize(image[self.h0:self.h1, self.w0:self.w1], img_size, interpolation=cv2.INTER_LINEAR)
        return image


def getloader(args, labels, inframe, outframe):
    batch_size = {}
    for label in labels:
        batch_size[label] = args.batch_size
    train_dataset = Datasetmix(args.data_path, args.info_path, labels, MODE.TRAIN, inframe, outframe,
                               (args.image_width, args.image_height),
                               (args.output_width, args.output_height),
                               batch_sizes=batch_size)
    shape_dataset = Datasetmix(args.data_path, args.info_path, labels, MODE.VALIDATION, inframe, outframe,
                               (args.image_width, args.image_height),
                               (args.output_width, args.output_height),
                               batch_sizes=batch_size)
    parameter_dataset = Datasetmix(args.data_path, args.info_path, labels, MODE.TEST, inframe, outframe,
                               (args.image_width, args.image_height),
                               (args.output_width, args.output_height),
                               batch_sizes=batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn_batch_mix,
                                               batch_size=1, shuffle=True,
                                               num_workers=args.prefetch,
                                               pin_memory=True, drop_last=True)
    shape_loader = torch.utils.data.DataLoader(shape_dataset, collate_fn=collate_fn_batch_mix,
                                               batch_size=1, shuffle=False,
                                               num_workers=args.prefetch,
                                               pin_memory=True, drop_last=True)
    parameter_loader = torch.utils.data.DataLoader(parameter_dataset, collate_fn=collate_fn_batch_mix,
                                               batch_size=1, shuffle=False,
                                               num_workers=args.prefetch,
                                               pin_memory=True, drop_last=True)

    return train_loader, shape_loader, parameter_loader


def collate_fn_batch_mix(batch):
    return batch[0]

