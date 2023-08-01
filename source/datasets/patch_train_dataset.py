import os
import random

import numpy as np
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms, RandomHorizontalFlip, RandomRotation, \
    RandomVerticalFlip, RandomAdjustSharpness, RandomAutocontrast, Resize, InterpolationMode

from source.datasets.anomaly_creator.anomaly_creator import AnomalyCreator


class PatchTrainDataset(Dataset):

    def __init__(self, image_paths, imagenet_dir, patch_size, image_size,
                 horizontal_flip=False, vertical_flip=False, rotate=False, rotations=None,
                 adjust_sharpness=True, auto_contrast=True, color_jitter=True, create_anomaly=True):

        self.rotations = rotations
        self.image_paths = image_paths
        self.images = []
        self.patch_size = patch_size
        self.image_size = image_size
        self.create_anomaly = create_anomaly
        self.len = len(self.image_paths)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.rotate = rotate

        self.__build_transforms(horizontal_flip, vertical_flip, color_jitter,
                                rotate, adjust_sharpness, auto_contrast)

        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            self.images.append(img)

        if self.create_anomaly:
            self.anomaly_creator = AnomalyCreator(patch_size, patch_size, self.mean, self.std, imagenet_dir,
                                                  method='dfc', dfc_anomaly_size='all')

    def __getitem__(self, index):
        img = self.images[index]
        transformed_img = self.augmentation_transform(img)

        if self.rotate:
            if random.random() < 0.5:
                degrees = random.choice(self.rotations)
                transformed_img = transformed_img.rotate(degrees)

        images_normal = []
        images_abnormal = []
        masks_normal = []
        masks_abnormal = []

        width, height = self.patch_size, self.patch_size

        for x in range(0, transformed_img.size[0], width):
            for y in range(0, transformed_img.size[1], height):
                patch = transformed_img.crop(box=(x, y, x + width, y + height))

                if self.create_anomaly:
                    img_normal, img_abnormal, mask_normal, mask_abnormal = \
                        self.anomaly_creator(patch)

                    abnormal_image_patch = Image.fromarray(img_abnormal)

                    normal_image_patch = self.transform(img_normal)
                    abnormal_image_patch = self.transform(abnormal_image_patch)

                    images_normal.append(normal_image_patch)
                    images_abnormal.append(abnormal_image_patch)
                    masks_normal.append(mask_normal)
                    masks_abnormal.append(mask_abnormal)
                else:
                    normal_image_patch = self.transform(patch)
                    images_normal.append(normal_image_patch)

        if self.create_anomaly:
            return images_normal, images_abnormal, masks_normal, masks_abnormal
        else:
            return images_normal

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

        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    # endregion
