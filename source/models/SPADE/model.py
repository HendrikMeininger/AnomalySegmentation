from typing import Tuple

import numpy as np
import torch
import os

from PIL import Image

from source.datasets.dataset import Dataset
from source.models.SPADE.base_spade.tester import Tester
from source.models.SPADE.base_spade.trainer import Trainer
from source.utils import visualization

"""
    Implementation of SPADE: Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    Base implementation: https://github.com/byungjae89/SPADE-pytorch
    Paper: https://arxiv.org/abs/2005.02357
"""


class SPADE(object):

    # region init

    def __init__(self, model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_path is None:
            self.trained = False
        else:
            self.model_path = model_path
            self.valid_model = self.__valid_model()
            self.trained = True

    # endregion

    # region public methods

    def train(self, dataset: Dataset, output_dir: str, debugging: bool = False, image_size: int = 256,
              batch_size: int = 32) -> None:
        trainer = Trainer(output_dir=output_dir,
                          dataset=dataset,
                          debugging=debugging,
                          batch_size=batch_size,
                          image_size=image_size)

        trainer.train()

    def eval(self, dataset: Dataset, debugging: bool = False, self_ensembling: bool = False,
             image_size: int = 256, mask_size: int = 1024,
             rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
             h_flip: bool = False, h_flip_rot_90: bool = False,
             h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
             integration_limit: float = 0.3, top_k: int = 5) -> None:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        tester = Tester(model_path=self.model_path, debugging=debugging,
                        image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                        rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                        h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                        h_flip_rot_270=h_flip_rot_270,
                        integration_limit=integration_limit, top_k=top_k)

        tester.evaluate(dataset=dataset)

    def display_predictions(self, dataset: Dataset, debugging: bool = False, self_ensembling: bool = False,
                            image_size: int = 256, mask_size: int = 1024,
                            rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
                            h_flip: bool = False, h_flip_rot_90: bool = False,
                            h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                            integration_limit: float = 0.3, top_k: int = 5) -> None:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        tester = Tester(model_path=self.model_path, debugging=debugging,
                        image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                        rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                        h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                        h_flip_rot_270=h_flip_rot_270,
                        integration_limit=integration_limit, top_k=top_k)

        tester.display_predictions(dataset=dataset)

    def predict(self, image_path: str, display_prediction: bool, debugging: bool = False,
                mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225),
                self_ensembling: bool = False, image_size: int = 256,
                rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
                h_flip: bool = False, h_flip_rot_90: bool = False,
                h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                top_k: int = 5) -> Tuple[np.array, np.array]:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        tester = Tester(model_path=self.model_path, debugging=debugging,
                        image_size=image_size, use_self_ensembling=self_ensembling,
                        rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                        h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                        h_flip_rot_270=h_flip_rot_270, top_k=top_k)

        score, binary_score = tester.predict(image_path=image_path, mean=mean, std=std)

        if display_prediction:
            original = Image.open(image_path).convert('RGB')

            visualization.display_images(img_list=[original, score, binary_score],
                                         titles=['original', 'score', 'binary_score'],
                                         cols=3)

        return score, binary_score

    # endregion

    # region private methods

    def __valid_model(self) -> bool:
        valid = True
        valid = valid and os.path.exists(os.path.join(self.model_path, "avgpool.npy"))
        valid = valid and os.path.exists(os.path.join(self.model_path, "layer_1.npy"))
        valid = valid and os.path.exists(os.path.join(self.model_path, "layer_2.npy"))
        valid = valid and os.path.exists(os.path.join(self.model_path, "layer_3.npy"))

        return valid

    # endregion
