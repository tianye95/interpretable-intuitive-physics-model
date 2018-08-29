import torch
import torch.nn
import torch.autograd
import torchvision
import numpy as np
import functions


class Alexnet(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(Alexnet, self).__init__()
        self.features = torchvision.models.alexnet(pretrained=pretrained).features

    def forward(self, x):
        return self.features(x)


class PhysicsModel(torch.nn.Module):
    def __init__(self, num_feature, passthrough):
        super(PhysicsModel, self).__init__()

        self.num_feature = num_feature
        self.passthrough = passthrough
        self.alexnet = torch.nn.Sequential(
            Alexnet(pretrained=True),
        )
        # encoder for four frames
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 64, 3, 1, 1),
            torch.nn.LeakyReLU()
        )
        self.encoder2 = torch.nn.Sequential(
            functions.LinearPermutate(3136, 2048),
            functions.LinearPermutate(2048, 1024),
            functions.LinearPermutate(1024, num_feature)
        )
        self.select = functions.SelectiveFilter(active=True, passthrough=passthrough)
        self.flownetwork = functions.FlowDecode(dim_feature=num_feature)

    def forward(self, inputs, indices_dict):
        extra_outputs = {}
        parameter_outputs = [self.alexnet(image) for key, image in inputs.items()]
        feature = torch.cat(parameter_outputs, dim=1)
        feature = self.encoder1(feature).view(feature.size(0), -1)
        feature = self.encoder2(feature)

        feature_original = feature.clone()
        feature = self.select(feature, indices_dict)

        extra_outputs["original_feature"] = feature_original
        extra_outputs["selected_feature"] = feature
        output, _ = self.flownetwork(inputs[3], feature)
        return output, extra_outputs


class InterpolationModel(torch.nn.Module):
    def __init__(self, num_feature, passthrough):
        super(InterpolationModel, self).__init__()

        self.num_feature = num_feature
        self.passthrough = passthrough
        self.alexnet = torch.nn.Sequential(
            Alexnet(pretrained=True),
        )
        # encoder for four frames
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 64, 3, 1, 1),
            torch.nn.LeakyReLU()
        )
        self.encoder2 = torch.nn.Sequential(
            functions.LinearPermutate(3136, 2048),
            functions.LinearPermutate(2048, 1024),
            functions.LinearPermutate(1024, num_feature)
        )
        self.select = functions.SelectiveFilter(active=True, passthrough=passthrough)
        self.flownetwork = functions.FlowDecode(dim_feature=num_feature)

    def forward(self, inputs, indices_dict):
        extra_outputs = {}
        parameter_outputs = [self.alexnet(image) for key, image in inputs.items()]
        feature = torch.cat(parameter_outputs, dim=1)
        feature = self.encoder1(feature).view(feature.size(0), -1)
        feature = self.encoder2(feature)

        feature_original = feature.clone()
        feature = self.select(feature, indices_dict)

        extra_outputs["original_feature"] = feature_original
        extra_outputs["selected_feature"] = feature
        extra_outputs['original'], _ = self.flownetwork(inputs[3], feature_original)

        # interpolation
        feature_test = feature_original.clone()
        input_test, feature_test = self.interpolate(feature_test, inputs[3], indices_dict)
        interpolation, _ = self.flownetwork(input_test, feature_test)

        return interpolation, extra_outputs

    def interpolate(self, feature, input, labels_and_indices):
        new_feature = feature.clone()
        new_input = input.clone()
        for label, indices_list in labels_and_indices.items():
            if label not in self.passthrough:
                print("label {} is not contained in passthrough".format(label))
                return feature, input
            for indices in indices_list:
                passthrough = torch.LongTensor(self.passthrough[label]).cuda()
                indices = torch.LongTensor(indices).cuda()
                label_data = torch.index_select(new_feature, 0, indices)
                data = torch.index_select(label_data, 1, passthrough)
                select_min, select_max = data[0, :].clone(), data[-1, :].clone()
                for i in range(5):
                    item = label_data[0, :].clone()
                    item[passthrough] = (1 - 0.25 * i) * select_min + 0.25 * i * select_max
                    new_feature[indices[i], :] = item

        for label, indices_list in labels_and_indices.items():
            for indices in indices_list:
                label_input = new_input[indices, :, :, :].clone()
                for i in range(5):
                    replace_input = label_input[0, :, :, :].clone()
                    new_input[indices[i], :, :, :] = replace_input
        return new_input, new_feature

