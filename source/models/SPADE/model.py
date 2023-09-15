import torch
import os

from source.datasets.dataset import Dataset
from source.models.SPADE.base_spade.trainer import Trainer

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
