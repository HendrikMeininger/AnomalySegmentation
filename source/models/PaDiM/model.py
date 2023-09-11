import torch
import os

from source.models.PaDiM.base_padim.trainer import Trainer
from source.models.PaDiM.patch_padim.trainer import Trainer as PatchTrainer

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

    def eval(self, dataset, debugging: bool = False) -> None:
        pass

    def display_predictions(self, dataset, debugging: bool = False) -> None:
        pass

    def predict(self, image_path: str, display_prediction: bool, debugging: bool = False) -> None:
        pass

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
