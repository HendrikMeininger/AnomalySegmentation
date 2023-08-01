import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from source.datasets.train_dataset import TrainDataset
from tools.PaDiM.padim import PaDiM


"""
    Implementation of PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
    Code modified from https://github.com/Pangoraw/PaDiM
    Paper: https://arxiv.org/abs/2011.08785
"""


class Trainer(object):

    def __init__(self, output_dir, image_paths, dataset_dir, batch_size=30, num_embeddings=130, backbone="resnet18",
                 image_size: int = 512):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.image_size = image_size

        self.padim = PaDiM(num_embeddings=num_embeddings, device=self.device, backbone=backbone,
                           size=(image_size, image_size))

    def train(self):
        train_data = TrainDataset(image_paths=self.image_paths, img_size=self.image_size,
                                  imagenet_dir="E:/imagenet/data", horizontal_flip=False, vertical_flip=False,
                                  rotate=False, adjust_sharpness=False, auto_contrast=False, color_jitter=False)
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.padim.train(train_loader)

        N, means, covs, embedding_ids = self.padim.get_residuals()

        self.__save_model(N, means, covs, embedding_ids)

    def __save_model(self, N, means, covs, embedding_ids):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        n_paths = join(self.output_dir, "n.npy")
        np.save(n_paths, N)

        means_paths = join(self.output_dir, "means.npy")
        np.save(means_paths, means)

        covs_paths = join(self.output_dir, "covs.npy")
        np.save(covs_paths, covs)

        embedding_ids_paths = join(self.output_dir, "embedding_ids.npy")
        np.save(embedding_ids_paths, embedding_ids)
