import os
import cv2
import numpy as np
from typing import Tuple
from os.path import join, isfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Resize, InterpolationMode


class TestDataset(Dataset):

    def __init__(self, path_to_dataset, image_size=256, mask_size=1024,
                 mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225)):
        self.path_to_dataset = path_to_dataset
        self.test_data_path = path_to_dataset + '/test'
        self.gt_data_path = path_to_dataset + '/ground_truth'

        self.image_size = image_size
        self.mask_size = mask_size
        self.mean = list(mean)
        self.std = list(std)

        self.masks, self.image_paths = self.__load_images_and_masks()
        self.len = len(self.image_paths)

        self.__build_transforms()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert('RGB')

        original = self.resize_transform(img)
        preprocessed = self.transform(original)

        mask = self.masks[index]
        mask = np.resize(mask, (self.mask_size, self.mask_size))

        return np.array(original), preprocessed, mask

    def __len__(self):
        return self.len

    def __load_images_and_masks(self):
        masks = []
        image_paths = []

        dirs = [f for f in os.listdir(self.test_data_path)]
        dirs.sort()

        for image_dir in dirs:
            img_dir = join(self.test_data_path, image_dir)
            mask_dir = join(self.gt_data_path, image_dir)

            img_paths = [f for f in os.listdir(img_dir) if isfile(join(img_dir, f))]
            img_paths.sort()
            for path in img_paths:
                image_paths.append(join(img_dir, path))

            if image_dir == 'good':
                for _ in img_paths:
                    mask = np.zeros(shape=(self.mask_size, self.mask_size))
                    masks.append(mask)
            else:
                mask_paths = [f for f in os.listdir(mask_dir) if isfile(join(mask_dir, f))]
                mask_paths.sort()
                for path in mask_paths:
                    mask = self.__load_mask(mask_dir, path)
                    masks.append(mask)

        return masks, image_paths

    def __load_image(self, image_dir, image_name):
        img = Image.open(join(image_dir, image_name)).convert('RGB')

        return img

    def __load_mask(self, mask_dir, mask_name):
        mask = cv2.imread(join(mask_dir, mask_name), cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.mask_size, self.mask_size))
        mask = mask / 255

        return mask

    def __build_transforms(self):
        resize_transform = Resize(size=(self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_transform = transforms.Compose([resize_transform])

        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
