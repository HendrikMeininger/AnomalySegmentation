import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from source.datasets.dataset import Dataset
from source.datasets.patch_train_dataset import PatchTrainDataset
from source.datasets.train_dataset import TrainDataset
from source.models.PaDiM.backbone.padim import PaDiM


class Trainer(object):

    # region init

    def __init__(self, output_dir: str, dataset: Dataset, batch_size: int = 32, num_embeddings=130,
                 backbone="resnet18", image_size: int = 512, debugging: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_dir = output_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.backbone = backbone
        self.debugging = debugging

        self.__init_models()

    def __init_models(self) -> None:
        self.padim_big = PaDiM(num_embeddings=self.num_embeddings, device=self.device, backbone=self.backbone,
                               size=(self.image_size, self.image_size))
        self.padim_medium = PaDiM(num_embeddings=self.num_embeddings, device=self.device, backbone=self.backbone,
                                  size=(self.image_size, self.image_size))
        self.padim_small = PaDiM(num_embeddings=self.num_embeddings, device=self.device, backbone=self.backbone,
                                 size=(self.image_size, self.image_size))

    # endregion

    # region public methods

    def train(self) -> None:
        if os.path.exists(os.path.join(self.output_dir, 'big', 'covs.npy')):
            self.__log("Big model already exists. Skipped training big model.")
        else:
            self.__train_big_model()
            self.__log("Finished training PaDiM model for big patches.")

        if os.path.exists(os.path.join(self.output_dir, 'medium', 'covs.npy')):
            self.__log("Medium model already exists. Skipped training medium model.")
        else:
            self.__train_medium_model()
            self.__log("Finished training PaDiM model for medium patches.")

        if os.path.exists(os.path.join(self.output_dir, 'small', 'covs.npy')):
            self.__log("Small model already exists. Skipped training small model.")
        else:
            self.__train_small_model()
            self.__log("Finished training PaDiM model for small patches.")

    # endregion

    # region private methods

    def __train_big_model(self) -> None:
        train_loader = self.dataset.get_train_dataloader(self.batch_size)
        self.padim_big.train(train_loader, epochs=3)

        N, means, covs, embedding_ids = self.padim_big.get_residuals()

        self.__save_model(N, means, covs, embedding_ids, dir_name="big")

    def __train_medium_model(self) -> None:
        patch_train_loader = self.dataset.get_medium_patches_train_dataloader(self.batch_size)
        self.padim_medium.train(patch_train_loader, epochs=3, use_patches=True)

        N, means, covs, embedding_ids = self.padim_medium.get_residuals()

        self.__save_model(N, means, covs, embedding_ids, dir_name="medium")

    def __train_small_model(self) -> None:
        patch_train_loader = self.dataset.get_small_patches_train_dataloader(self.batch_size)
        self.padim_small.train(patch_train_loader, epochs=3, use_patches=True)

        N, means, covs, embedding_ids = self.padim_small.get_residuals()

        self.__save_model(N, means, covs, embedding_ids, dir_name="small")

    def __save_model(self, N, means, covs, embedding_ids, dir_name: str) -> None:
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

    def __log(self, message: str) -> None:
        if self.debugging:
            print(message)

    # endregion
