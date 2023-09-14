import math
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from os.path import join
from typing import List
from PIL import Image
from torchvision.transforms import transforms, Resize

from source.models.PaDiM.backbone.padim import PaDiM
from source.models.utils import BaseTester


class Tester(BaseTester):

    # region init

    def __init__(self, model_path: str, debugging: bool = False,
                 image_size: int = 1024, mask_size: int = 1024, patch_size: int = 256,
                 use_self_ensembling: bool = False,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 integration_limit: float = 0.3, backbone: str = "resnet18"):
        self.patch_size = patch_size
        self.resize_transform = Resize(size=patch_size, interpolation=TF.InterpolationMode.BILINEAR)

        super().__init__(model_path=model_path, debugging=debugging, image_size=image_size, mask_size=mask_size,
                         use_self_ensembling=use_self_ensembling, rot_90=rot_90, rot_180=rot_180, rot_270=rot_270,
                         h_flip=h_flip, h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                         h_flip_rot_270=h_flip_rot_270, integration_limit=integration_limit)
        self.backbone = backbone

        self.__load_model()

    def __load_model(self):
        self.padim_big = self.__load_patch_model(join(self.model_path, "big"))
        self.padim_medium = self.__load_patch_model(join(self.model_path, "medium"))
        self.padim_small = self.__load_patch_model(join(self.model_path, "small"))

    def __load_patch_model(self, model_path):
        n = np.load(join(model_path, "n.npy")).item()
        means = np.load(join(model_path, "means.npy"))
        covs = np.load(join(model_path, "covs.npy"))
        embedding_ids = np.load(join(model_path, "embedding_ids.npy"))
        padim = PaDiM.from_residuals(N=n,
                                     means=means,
                                     covs=covs,
                                     embedding_ids=embedding_ids,
                                     backbone=self.backbone,
                                     device=self.device,
                                     img_size=self.image_size)
        return padim

    # endregion

    # region implement abstract methods

    def score(self, img_input) -> np.array:
        big_patches_score = self.__score_big_patches(img_input)
        medium_patches_score = self.__score_medium_patches(img_input)
        small_patches_score = self.__score_small_patches(img_input)

        final_score = self.__calc_final_score(big_patches_score, medium_patches_score, small_patches_score)

        return final_score

    def score_with_augmentation(self, img_input) -> np.array:
        score_list = self.__get_self_ensembling_scores(img_input)
        final_score = self.__combine_scores(score_list)

        return final_score

    def preprocess_img(self, image_path: str, mean: List[float], std: List[float]):
        original = Image.open(image_path).convert('RGB')

        normalize = transforms.Normalize(mean=mean, std=std)
        resize = torchvision.transforms.Resize(size=self.patch_size, interpolation=TF.InterpolationMode.BILINEAR)

        transform = transforms.Compose([transforms.ToTensor(), normalize, resize])
        preprocessed = transform(original)

        preprocessed = preprocessed[None, :]

        return preprocessed

    # endregion

    # region private methods

    def __calc_final_score(self, big, medium, small) -> np.array:
        # score = np.max(big, np.max(medium, small))
        score = (big + medium + small) / 3
        return score

    def __score_big_patches(self, img) -> np.array:
        return self.__score_patch(img, self.padim_big, 1024)

    def __score_medium_patches(self, img) -> np.array:
        score = np.zeros(shape=(self.mask_size, self.mask_size), dtype=float)

        width, height = 512, 512

        for x in range(0, 1024, width):
            for y in range(0, 1024, height):
                patch = img[:, :, x:x + width, y:y + height]

                patch_score = self.__score_patch(patch, self.padim_medium, 512)
                score_x = x
                score_y = y
                score[score_x:score_x + width, score_y:score_y + height] = patch_score

        return score

    def __score_small_patches(self, img) -> np.array:
        score = np.zeros(shape=(self.mask_size, self.mask_size), dtype=float)

        width, height = 256, 256

        for x in range(0, 1024, width):
            for y in range(0, 1024, height):
                patch = img[:, :, x:x + width, y:y + height]

                patch_score = self.__score_patch(patch, self.padim_small, 256)
                score_x = x
                score_y = y
                score[score_x:score_x + width, score_y:score_y + height] = patch_score

        return score

    def __score_patch(self, patch, model, out_size) -> np.array:
        img = self.resize_transform(patch).to(self.device)

        distances = model.predict(img)
        w = int(math.sqrt(distances.numel()))
        raw_score = distances.reshape(1, 1, w, w)
        raw_score = F.interpolate(raw_score, size=(out_size, out_size), mode="bilinear",
                                  align_corners=True)

        raw_score = raw_score.detach().cpu().numpy().squeeze()

        return raw_score

    # end region
