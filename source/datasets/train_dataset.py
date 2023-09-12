import random
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from source.datasets.anomaly_creator.anomaly_creator import AnomalyCreator

"""
    Dataset to train DFC SPADE or PaDiM model
    Data can be augmented
"""


class TrainDataset(Dataset):

    def __init__(self, image_paths: List[str], imagenet_dir: str, img_size: int = 256, mask_size: int = 1024,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 self_supervised_training: bool = True,
                 mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225),
                 dfc_anomaly_size: str = 'big', method: str = 'dfc', cutpaste_mode: str = 'all'):

        self.image_paths = image_paths
        self.images = []
        self.normal_images = []
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

        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            self.images.append(img)

        self.__build_transforms()

        if self.self_supervised_training:
            self.anomaly_creator = AnomalyCreator(img_size=img_size, mask_size=mask_size, mean=self.mean, std=self.std,
                                                  imagenet_dir=imagenet_dir, method=method,
                                                  dfc_anomaly_size=dfc_anomaly_size, cutpaste_mode=cutpaste_mode)

    def __getitem__(self, index):
        img = self.images[index]
        augmented_img = self.__augment_img(img)

        if self.self_supervised_training:
            img_normal, img_abnormal, mask_normal, mask_abnormal = \
                self.anomaly_creator(augmented_img)

            img_abnormal = Image.fromarray(img_abnormal)

            img_normal = self.transform(img_normal)
            img_abnormal = self.transform(img_abnormal)

            return img_normal, img_abnormal, mask_normal, mask_abnormal
        else:
            return self.transform(augmented_img)

    def __len__(self):
        return self.len

    # region private methods

    def __build_transforms(self):
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
