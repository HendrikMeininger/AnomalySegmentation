import math
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join
from source.models.PaDiM.backbone.padim import PaDiM
from source.models.utils import BaseTester


"""
    Implementation of PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
    Code modified from https://github.com/Pangoraw/PaDiM
    Paper: https://arxiv.org/abs/2011.08785
"""


class Tester(BaseTester):

    # region init

    def __init__(self, model_path: str, debugging: bool = False,
                 image_size: int = 256, mask_size: int = 1024, use_self_ensembling: bool = False,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 integration_limit: float = 0.3, backbone: str = "wide_resnet50"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(model_path=model_path, debugging=debugging, image_size=image_size, mask_size=mask_size,
                         use_self_ensembling=use_self_ensembling, rot_90=rot_90, rot_180=rot_180, rot_270=rot_270,
                         h_flip=h_flip, h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                         h_flip_rot_270=h_flip_rot_270, integration_limit=integration_limit)
        self.backbone = backbone

        self.__load_model()

    def __load_model(self) -> None:
        n = np.load(join(self.model_path, "n.npy")).item()
        means = np.load(join(self.model_path, "means.npy"))
        covs = np.load(join(self.model_path, "covs.npy"))
        embedding_ids = np.load(join(self.model_path, "embedding_ids.npy"))
        padim = PaDiM.from_residuals(N=n,
                                     means=means,
                                     covs=covs,
                                     embedding_ids=embedding_ids,
                                     backbone=self.backbone,
                                     device=self.device,
                                     img_size=self.image_size)
        self.padim = padim

    # end region

    # region private methods

    def __score(self, img_input) -> np.array:  # returns score with shape (1024, 1024)
        distances = self.padim.predict(img_input)
        w = int(math.sqrt(distances.numel()))
        raw_score = distances.reshape(1, 1, w, w)
        raw_score = F.interpolate(raw_score, size=(self.mask_size, self.mask_size), mode="bilinear",
                                  align_corners=True)

        raw_score = raw_score.detach().cpu().numpy().squeeze()

        return raw_score

    def __score_with_augmentation(self, img_input) -> np.array:
        score_list = self.__get_self_ensembling_scores(img_input)
        final_score = self.__combine_scores(score_list)

        return final_score

    # endregion
