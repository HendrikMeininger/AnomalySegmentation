import os
import pickle
from collections import OrderedDict
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from source.datasets import train_dataset
from source.datasets.train_dataset import TrainDataset
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

"""
    Implementation of Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    Code modified from https://github.com/byungjae89/SPADE-pytorch/tree/master
    Paper: https://arxiv.org/abs/2005.02357
"""


class Trainer(object):

    def __init__(self, path_to_dataset, output_dir, batch_size=32, image_size=256):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.path_to_dataset = path_to_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.__init_model()

    def __init_model(self):
        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
        self.model.to(self.device)
        self.model.eval()

    def train(self):
        image_paths = train_dataset.get_train_img_paths(self.path_to_dataset)
        train_data = TrainDataset(image_paths=image_paths, img_size=self.image_size,
                                  imagenet_dir="E:/imagenet/data", horizontal_flip=True, vertical_flip=False,
                                  rotate=False, adjust_sharpness=False, auto_contrast=False, color_jitter=False,
                                  anomaly_size='all', method='all')
        train_loader = DataLoader(dataset=train_data, batch_size=32, pin_memory=True)

        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.model.avgpool.register_forward_hook(hook)

        train_outputs = [[], [], [], []]

        for img_normal, img_abnormal, mask_normal, mask_abnormal in train_loader:
            img_normal = img_normal.to(self.device)
            # model prediction
            with torch.no_grad():
                pred = self.model(img_normal)
                del pred
                del img_normal
            # get intermediate layer outputs
            for i in range(len(train_outputs)):
                train_outputs[i].append(outputs[i])
            # train_outputs = [lst + [outputs[i]] for i, lst in enumerate(train_outputs)]

            # initialize hook outputs
            outputs = []
        for i in range(len(train_outputs)):
            train_outputs[i] = np.array(torch.cat(train_outputs[i], 0).cpu())
        # train_outputs = [np.array(torch.stack(lst, dim=0).cpu()) for lst in train_outputs]

        self.__save_model(train_outputs=train_outputs)

    def __save_model(self, train_outputs):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        l1_path = join(self.output_dir, "layer_1.npy")
        np.save(l1_path, train_outputs[0])

        l2_path = join(self.output_dir, "layer_2.npy")
        np.save(l2_path, train_outputs[1])

        l3_path = join(self.output_dir, "layer_3.npy")
        np.save(l3_path, train_outputs[2])

        pool_path = join(self.output_dir, "avgpool.npy")
        np.save(pool_path, train_outputs[3])
