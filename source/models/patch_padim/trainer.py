import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from source.datasets.patch_train_dataset import PatchTrainDataset
from source.datasets.train_dataset import TrainDataset
from source.models.padim_backbone.padim import PaDiM


class Trainer(object):

    # region init

    def __init__(self, output_dir, image_paths, dataset_dir, batch_size=30, num_embeddings=130, backbone="resnet18",
                 image_size: int = 512):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.backbone = backbone

        self.__init_models()

    def __init_models(self):
        self.padim_big = PaDiM(num_embeddings=self.num_embeddings, device=self.device, backbone=self.backbone,
                               size=(self.image_size, self.image_size))
        self.padim_medium = PaDiM(num_embeddings=self.num_embeddings, device=self.device, backbone=self.backbone,
                                  size=(self.image_size, self.image_size))
        self.padim_small = PaDiM(num_embeddings=self.num_embeddings, device=self.device, backbone=self.backbone,
                                 size=(self.image_size, self.image_size))

    # endregion

    # region public methods

    def train(self):
        if os.path.exists(os.path.join(self.output_dir, 'big', 'covs.npy')):
            print("Big model already exists. Skipped training big model.")
        else:
            self.__train_big_model()
            print("Finished training PaDiM model for big patches.")

        if os.path.exists(os.path.join(self.output_dir, 'medium', 'covs.npy')):
            print("Medium model already exists. Skipped training medium model.")
        else:
            self.__train_medium_model()
            print("Finished training PaDiM model for medium patches.")

        if os.path.exists(os.path.join(self.output_dir, 'small', 'covs.npy')):
            print("Small model already exists. Skipped training small model.")
        else:
            self.__train_small_model()
            print("Finished training PaDiM model for small patches.")

    # endregion

    # region private methods

    def __train_big_model(self):
        train_data = TrainDataset(image_paths=self.image_paths, img_size=self.image_size,
                                  imagenet_dir="", horizontal_flip=False, vertical_flip=False,
                                  rotate=False, adjust_sharpness=False, auto_contrast=False, color_jitter=False)
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.padim_big.train(train_loader, epochs=3)

        N, means, covs, embedding_ids = self.padim_big.get_residuals()

        self.__save_model(N, means, covs, embedding_ids, dir_name="big")

    def __train_medium_model(self):
        patch_train_data = PatchTrainDataset(image_paths=self.image_paths, patch_size=256, image_size=512,
                                             imagenet_dir="",
                                             horizontal_flip=True, vertical_flip=True, rotate=True,
                                             adjust_sharpness=False, auto_contrast=False, color_jitter=False,
                                             create_anomaly=False)
        train_loader = DataLoader(dataset=patch_train_data, batch_size=self.batch_size, shuffle=True)
        self.padim_medium.train(train_loader, epochs=3, use_patches=True)

        N, means, covs, embedding_ids = self.padim_medium.get_residuals()

        self.__save_model(N, means, covs, embedding_ids, dir_name="medium")

    def __train_small_model(self):
        patch_train_data = PatchTrainDataset(image_paths=self.image_paths, patch_size=256, image_size=1024,
                                             imagenet_dir="",
                                             horizontal_flip=True, vertical_flip=True, rotate=True,
                                             adjust_sharpness=False, auto_contrast=False, color_jitter=False,
                                             create_anomaly=False)
        train_loader = DataLoader(dataset=patch_train_data, batch_size=self.batch_size, shuffle=True)
        self.padim_small.train(train_loader, epochs=3, use_patches=True)

        N, means, covs, embedding_ids = self.padim_small.get_residuals()

        self.__save_model(N, means, covs, embedding_ids, dir_name="small")

    def __save_model(self, N, means, covs, embedding_ids, dir_name):
        model_dir = join(self.output_dir, dir_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        n_paths = join(model_dir, "n.npy")
        np.save(n_paths, N)

        means_paths = join(model_dir, "means.npy")
        np.save(means_paths, means)

        covs_paths = join(model_dir, "covs.npy")
        np.save(covs_paths, covs)

        embedding_ids_paths = join(model_dir, "embedding_ids.npy")
        np.save(embedding_ids_paths, embedding_ids)

    # endregion
