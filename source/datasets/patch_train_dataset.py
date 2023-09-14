import random
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from source.datasets.anomaly_creator.anomaly_creator import AnomalyCreator

"""
    Dataset to train DFC SPADE or PaDiM model with patches
    Data can be augmented
"""


class PatchTrainDataset(Dataset):

    def __init__(self, image_paths: List[str], imagenet_dir: str,
                 patch_size: int, img_size: int = 256, mask_size: int = 1024,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 self_supervised_training: bool = True,
                 mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225),
                 dfc_anomaly_size: str = 'big', method: str = 'dfc', cutpaste_mode: str = 'all'):

        self.image_paths = image_paths
        self.images = []
        self.normal_images = []
        self.patch_size = patch_size
        self.img_size = img_size
        self.mask_size = mask_size
        self.len = len(self.image_paths)
        self.mean = list(mean)
        self.std = list(std)

        self.rot_90 = rot_90
        self.rot_180 = rot_180
        self.rot_270 = rot_270
        self.h_flip = h_flip
        self.h_flip_rot_90 = h_flip_rot_90
        self.h_flip_rot_180 = h_flip_rot_180
        self.h_flip_rot_270 = h_flip_rot_270

        self.self_supervised_training = self_supervised_training

        self.__build_transforms()
        self.__build_aug_transforms()

        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            self.images.append(img)

        if self.self_supervised_training:
            self.anomaly_creator = AnomalyCreator(img_size=patch_size, mask_size=mask_size, mean=self.mean, std=self.std,
                                                  imagenet_dir=imagenet_dir, method=method,
                                                  dfc_anomaly_size=dfc_anomaly_size, cutpaste_mode=cutpaste_mode)

    def __getitem__(self, index):
        img = self.images[index]
        augmented_img = self.__augment_img(img)

        images_normal = []
        images_abnormal = []
        masks_normal = []
        masks_abnormal = []

        width, height = self.patch_size, self.patch_size

        for x in range(0, augmented_img.size[0], width):
            for y in range(0, augmented_img.size[1], height):
                patch = augmented_img.crop(box=(x, y, x + width, y + height))

                if self.self_supervised_training:
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

        if self.self_supervised_training:
            return images_normal, images_abnormal, masks_normal, masks_abnormal
        else:
            return images_normal

    def __len__(self):
        return self.len

    # region private methods

    def __build_transforms(self) -> None:
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def __build_aug_transforms(self) -> None:
        self.possible_transforms: List[transforms.Compose] = [transforms.Compose([])]

        if self.rot_90:
            trans = transforms.Compose([transforms.RandomRotation(degrees=[90, 90])])
            self.possible_transforms.append(trans)
        if self.rot_180:
            trans = transforms.Compose([transforms.RandomRotation(degrees=[180, 180])])
            self.possible_transforms.append(trans)
        if self.rot_270:
            trans = transforms.Compose([transforms.RandomRotation(degrees=[270, 270])])
            self.possible_transforms.append(trans)
        if self.h_flip:
            trans = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
            self.possible_transforms.append(trans)
        if self.h_flip_rot_90:
            trans = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                        transforms.RandomRotation(degrees=[90, 90])])
            self.possible_transforms.append(trans)
        if self.h_flip_rot_180:
            trans = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                        transforms.RandomRotation(degrees=[180, 180])])
            self.possible_transforms.append(trans)
        if self.h_flip_rot_270:
            trans = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                        transforms.RandomRotation(degrees=[270, 270])])
            self.possible_transforms.append(trans)

    def __augment_img(self, img: Image) -> Image:
        aug_value: int = int(random.uniform(0, len(self.possible_transforms)))
        augmented = self.possible_transforms[aug_value](img)

        return augmented

    # endregion
