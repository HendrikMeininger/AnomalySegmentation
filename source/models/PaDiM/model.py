import torch
import os
import numpy as np

from PIL import Image
from typing import Tuple
from source.datasets.dataset import Dataset
from source.models.PaDiM.base_padim.trainer import Trainer
from source.models.PaDiM.patch_padim.trainer import Trainer as PatchTrainer
from source.models.PaDiM.base_padim.tester import Tester
from source.models.PaDiM.patch_padim.tester import Tester as PatchTester
from source.utils import visualization

"""
    Implementation of PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
    Code modified from https://github.com/Pangoraw/PaDiM
    Paper: https://arxiv.org/abs/2011.08785
"""


class PaDiM(object):

    # region init

    def __init__(self, model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_path is None:
            self.trained = False
        else:
            self.valid_model = self.__check_model(model_path)
            self.trained = True
            self.model_path = model_path

    def __check_model(self, model_path: str) -> bool:
        if self.__valid_model(model_path):
            self.use_patches = False
            return True
        elif self.__valid_patch_model(model_path):
            self.use_patches = True
            return True

        return False

    # endregion

    # region public methods

    def train(self, dataset, output_dir: str, debugging: bool = False, num_embeddings=130, backbone="resnet18",
              image_size: int = 512) -> None:
        if self.use_patches:
            trainer = PatchTrainer(output_dir=output_dir,
                                   dataset=dataset,
                                   num_embeddings=num_embeddings,
                                   backbone=backbone,
                                   image_size=image_size,
                                   debugging=debugging)
        else:
            trainer = Trainer(output_dir=output_dir,
                              dataset=dataset,
                              num_embeddings=num_embeddings,
                              backbone=backbone,
                              image_size=image_size,
                              debugging=debugging)

        trainer.train()

    def eval(self, dataset: Dataset, debugging: bool = False, self_ensembling: bool = False,
             image_size: int = 256, mask_size: int = 1024,
             rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
             h_flip: bool = False, h_flip_rot_90: bool = False,
             h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
             integration_limit: float = 0.3, backbone: str = 'wide_resnet50') -> None:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        if self.use_patches:
            tester = PatchTester(model_path=self.model_path, debugging=debugging,
                                 image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                                 rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                                 h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                                 h_flip_rot_270=h_flip_rot_270,
                                 integration_limit=integration_limit, backbone=backbone)
        else:
            tester = Tester(model_path=self.model_path, debugging=debugging,
                            image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                            rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                            h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                            h_flip_rot_270=h_flip_rot_270,
                            integration_limit=integration_limit, backbone=backbone)

        tester.evaluate(dataset=dataset)

    def display_predictions(self, dataset: Dataset, debugging: bool = False, self_ensembling: bool = False,
                            image_size: int = 256, mask_size: int = 1024,
                            rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
                            h_flip: bool = False, h_flip_rot_90: bool = False,
                            h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                            backbone: str = 'wide_resnet50') -> None:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        if self.use_patches:
            tester = PatchTester(model_path=self.model_path, debugging=debugging,
                                 image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                                 rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                                 h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                                 h_flip_rot_270=h_flip_rot_270, backbone=backbone)
        else:
            tester = Tester(model_path=self.model_path, debugging=debugging,
                            image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                            rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                            h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                            h_flip_rot_270=h_flip_rot_270, backbone=backbone)

        tester.display_predictions(dataset=dataset)

    def predict(self, image_path: str, display_prediction: bool, debugging: bool = False,
                mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225),
                self_ensembling: bool = False, image_size: int = 256,
                rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
                h_flip: bool = False, h_flip_rot_90: bool = False,
                h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                backbone: str = 'wide_resnet50') -> Tuple[np.array, np.array]:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        if self.use_patches:
            tester = PatchTester(model_path=self.model_path, debugging=debugging,
                                 image_size=image_size, use_self_ensembling=self_ensembling,
                                 rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                                 h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                                 h_flip_rot_270=h_flip_rot_270, backbone=backbone)
        else:
            tester = Tester(model_path=self.model_path, debugging=debugging,
                            image_size=image_size, use_self_ensembling=self_ensembling,
                            rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                            h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                            h_flip_rot_270=h_flip_rot_270, backbone=backbone)

        score, binary_score = tester.predict(image_path=image_path, mean=mean, std=std)

        if display_prediction:
            original = Image.open(image_path).convert('RGB')

            visualization.display_images(img_list=[original, score, binary_score],
                                         titles=['original', 'score', 'binary_score'],
                                         cols=3)

        return score, binary_score

    # endregion

    # region private methods

    def __valid_model(self, model_path: str) -> bool:
        valid = True
        valid = valid and os.path.exists(os.path.join(model_path, "covs.npy"))
        valid = valid and os.path.exists(os.path.join(model_path, "embedding_ids.npy"))
        valid = valid and os.path.exists(os.path.join(model_path, "means.npy"))
        valid = valid and os.path.exists(os.path.join(model_path, "n.npy"))

        return valid

    def __valid_patch_model(self, model_path: str) -> bool:
        valid = True
        valid = valid and self.__valid_model(os.path.join(model_path, "big"))
        valid = valid and self.__valid_model(os.path.join(model_path, "medium"))
        valid = valid and self.__valid_model(os.path.join(model_path, "small"))

        return valid

    # endregion
