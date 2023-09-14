import os
from typing import List, Tuple
from torch.utils.data import DataLoader

from source.datasets.test_dataset import TestDataset
from source.datasets.train_dataset import TrainDataset
from source.datasets.patch_train_dataset import PatchTrainDataset


class Dataset(object):

    # region init

    def __init__(self, path_to_dataset: str, img_size: int = 256, mask_size: int = 1024, imagenet_dir: str = None,
                 self_supervised_training: bool = False,
                 mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225),
                 anomaly_creation_method='dfc', dfc_anomaly_size='big', cutpaste_mode='all',
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False):
        self.path_to_dataset = path_to_dataset
        self.img_size = img_size
        self.mask_size = mask_size
        self.imagenet_dir = imagenet_dir
        self.self_supervised_training = self_supervised_training
        self.mean = mean
        self.std = std

        self.anomaly_creation_method = anomaly_creation_method
        self.dfc_anomaly_size = dfc_anomaly_size
        self.cutpaste_mode = cutpaste_mode

        self.rot_90 = rot_90
        self.rot_180 = rot_180
        self.rot_270 = rot_270
        self.h_flip = h_flip
        self.h_flip_rot_90 = h_flip_rot_90
        self.h_flip_rot_180 = h_flip_rot_180
        self.h_flip_rot_270 = h_flip_rot_270

    # endregion

    # region public methods

    def get_train_dataloader(self, batch_size: int) -> DataLoader:
        image_paths = self.__get_train_img_paths()
        train_data = TrainDataset(image_paths=image_paths, imagenet_dir=self.imagenet_dir, img_size=self.img_size,
                                  mask_size=self.mask_size, rot_90=self.rot_90, rot_180=self.rot_180,
                                  rot_270=self.rot_270, h_flip=self.h_flip, h_flip_rot_90=self.h_flip_rot_90,
                                  h_flip_rot_180=self.h_flip_rot_180, h_flip_rot_270=self.h_flip_rot_270,
                                  self_supervised_training=self.self_supervised_training, mean=self.mean,
                                  std=self.std, dfc_anomaly_size=self.dfc_anomaly_size,
                                  method=self.anomaly_creation_method, cutpaste_mode=self.cutpaste_mode)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        return train_loader

    def get_medium_patches_train_dataloader(self, batch_size: int) -> DataLoader:
        return self.__get_patches_dataloader(batch_size=batch_size, img_size=self.img_size * 2)

    def get_small_patches_train_dataloader(self, batch_size: int) -> DataLoader:
        return self.__get_patches_dataloader(batch_size=batch_size, img_size=self.img_size * 4)

    def get_test_dataloader(self) -> DataLoader:
        test_dataset = TestDataset(path_to_dataset=self.path_to_dataset, mask_size=self.mask_size,
                                   image_size=self.img_size, mean=self.mean, std=self.std)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        return test_loader

    # endregion

    # region private methods

    def __get_patches_dataloader(self, batch_size: int, img_size: int):
        image_paths = self.__get_train_img_paths()
        patch_train_data = PatchTrainDataset(image_paths=image_paths, imagenet_dir=self.imagenet_dir,
                                             patch_size=self.img_size, img_size=img_size,
                                             mask_size=self.mask_size, rot_90=self.rot_90, rot_180=self.rot_180,
                                             rot_270=self.rot_270, h_flip=self.h_flip, h_flip_rot_90=self.h_flip_rot_90,
                                             h_flip_rot_180=self.h_flip_rot_180, h_flip_rot_270=self.h_flip_rot_270,
                                             self_supervised_training=self.self_supervised_training,
                                             mean=self.mean, std=self.std, dfc_anomaly_size=self.dfc_anomaly_size,
                                             method=self.anomaly_creation_method, cutpaste_mode=self.cutpaste_mode)
        train_loader = DataLoader(dataset=patch_train_data, batch_size=batch_size, shuffle=True)

        return train_loader

    def __get_train_img_paths(self) -> List[str]:
        train_data_path = os.path.join(self.path_to_dataset, 'train/good')
        image_paths = []

        for root, dirs, files in os.walk(train_data_path):
            for file in files:
                image_paths.append(os.path.join(root, file))

        return image_paths

    # endregion
