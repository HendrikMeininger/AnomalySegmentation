import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from source.datasets.train_dataset import TrainDataset
from source.models.PaDiM.backbone.padim import PaDiM


class Trainer(object):

    def __init__(self, output_dir, dataset, num_embeddings=130, backbone="resnet18",
                 image_size: int = 512, debugging: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_dir = output_dir
        self.dataset = dataset
        self.image_size = image_size
        self.debugging = debugging

        self.padim = PaDiM(num_embeddings=num_embeddings, device=self.device, backbone=backbone,
                           size=(image_size, image_size))

    def train(self):
        self.padim.train(self.dataset.get_train_dataloader())
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
