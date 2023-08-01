import os
import random

import numpy as np
from typing import List

import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision.transforms import transforms, RandomHorizontalFlip, RandomRotation, \
    RandomVerticalFlip, RandomAdjustSharpness, RandomAutocontrast, Resize, InterpolationMode

from source.datasets.anomaly_creator.anomaly_creator import AnomalyCreator
from source.utils import visualization

"""
Dataset to train DFC model
Data can be augmented
"""


def get_train_img_paths(path_to_dataset: str) -> List[str]:
    train_data_path = path_to_dataset + '/train/good'
    image_paths = []

    for root, dirs, files in os.walk(train_data_path):
        for file in files:
            image_paths.append(os.path.join(root, file))

    return image_paths


class TrainDataset(Dataset):

    def __init__(self, image_paths, imagenet_dir, img_size=256, mask_size=1024,
                 horizontal_flip=False, vertical_flip=False, rotate=False, adjust_sharpness=False,
                 auto_contrast=False, color_jitter=False, anomaly_size='all', method='all'):

        self.image_paths = image_paths
        self.images = []
        self.normal_images = []
        self.img_size = img_size
        self.mask_size = mask_size
        self.len = len(self.image_paths)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.rotate = rotate

        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            self.images.append(img)

        self.__build_transforms(horizontal_flip, vertical_flip, color_jitter,
                                rotate, adjust_sharpness, auto_contrast)

        self.anomaly_creator = AnomalyCreator(img_size, mask_size, self.mean, self.std, imagenet_dir,
                                              dfc_anomaly_size=anomaly_size, method=method, cutpaste_mode='all')

    def __getitem__(self, index):
        # image_path = self.image_paths[index]
        # img = Image.open(image_path)
        img = self.images[index]
        transformed_img = self.augmentation_transform(img)

        if self.rotate:
            if random.random() < 0.5:
                degrees = random.choice([90, 180, 270])
                # degrees = 180
                transformed_img = transformed_img.rotate(degrees)

        img_normal, img_abnormal, mask_normal, mask_abnormal = \
            self.anomaly_creator(transformed_img)

        img_abnormal = Image.fromarray(img_abnormal)

        img_normal = self.transform(img_normal)
        img_abnormal = self.transform(img_abnormal)

        return img_normal, img_abnormal, mask_normal, mask_abnormal

    def __len__(self):
        return self.len

    # region private methods

    def __build_transforms(self, horizontal_flip, vertical_flip, color_jitter,
                           rotate, adjust_sharpness, auto_contrast):
        self.augmentation_transform = transforms.Compose([])

        if horizontal_flip:
            random_flip = RandomHorizontalFlip(0.5)
            self.augmentation_transform.transforms.append(random_flip)
        if vertical_flip:
            random_flip = RandomVerticalFlip(0.5)
            self.augmentation_transform.transforms.append(random_flip)
        """if rotate:
            random_rotation = RandomRotation(degrees=(-2, 2))
            self.augmentation_transform.transforms.append(random_rotation)
        if adjust_sharpness:
            random_sharpness = RandomAdjustSharpness(sharpness_factor=2, p=0.5)
            self.augmentation_transform.transforms.append(random_sharpness)
        if auto_contrast:
            random_contrast = RandomAutocontrast(p=0.5)
            self.augmentation_transform.transforms.append(random_contrast)
        if color_jitter:
            random_color_jitter = transforms.ColorJitter(brightness=(0.5, 1.5),
                                                         contrast=1,
                                                         saturation=(0.5, 1.5),
                                                         hue=(-0.1, 0.1))
            self.augmentation_transform.transforms.append(random_color_jitter)"""

        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    # endregion
