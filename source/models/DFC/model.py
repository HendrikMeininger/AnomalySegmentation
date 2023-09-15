import os
import numpy as np
import torch
from PIL import Image
from typing import Tuple
from typing_extensions import Literal

from source.datasets.dataset import Dataset
from source.utils import visualization
from source.models.DFC.base_dfc.tester import Tester
from source.models.DFC.patch_dfc.tester import Tester as PatchTester
from source.models.DFC.base_dfc.trainer import Trainer
from source.models.DFC.patch_dfc.trainer import Trainer as PatchTrainer

"""
    Implementation of DFC: Learning deep feature correspondence for unsupervised anomaly detection and segmentation
    Base implementation: https://github.com/YoungGod/DFC
    Paper: https://www.sciencedirect.com/science/article/abs/pii/S0031320322003557
"""


class DFC(object):

    # region init

    _DATASET_TYPES = Literal["textures", "objects"]

    def __init__(self, model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_patches = None

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

    def train(self, dataset: Dataset, dataset_type: _DATASET_TYPES, output_dir: str, use_patches: bool = False,
              debugging: bool = False, batch_size: int = 8, epochs: int = 301, lr: float = 2e-4,
              train_split: float = 0.9, pretrained_weights_dir: str = None, imagenet_dir: str = None,
              early_stopping: bool = True, patience: int = 10) -> None:
        self.use_patches = use_patches

        if self.use_patches:
            trainer = PatchTrainer(output_dir=output_dir, dataset=dataset, dataset_type=dataset_type,
                                   batch_size=batch_size, n_epochs=epochs, lr=lr, train_split=train_split,
                                   pretrained_weights_dir=pretrained_weights_dir, imagenet_dir=imagenet_dir,
                                   early_stopping=early_stopping, patience=patience, debugging=debugging)
        else:
            trainer = Trainer(output_dir=output_dir, dataset=dataset, dataset_type=dataset_type, batch_size=batch_size,
                              n_epochs=epochs, lr=lr, train_split=train_split,
                              pretrained_weights_dir=pretrained_weights_dir, imagenet_dir=imagenet_dir,
                              early_stopping=early_stopping, patience=patience, debugging=debugging)

        trainer.train()

    def eval(self, dataset: Dataset, dataset_type: _DATASET_TYPES, pretrained_weights_dir: str = None,
             debugging: bool = False, self_ensembling: bool = False,
             image_size: int = 256, mask_size: int = 1024,
             rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
             h_flip: bool = False, h_flip_rot_90: bool = False,
             h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
             integration_limit: float = 0.3):
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        if self.use_patches:
            tester = PatchTester(model_path=self.model_path, debugging=debugging, dataset_type=dataset_type,
                                 pretrained_weights_dir=pretrained_weights_dir,
                                 image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                                 rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                                 h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                                 h_flip_rot_270=h_flip_rot_270,
                                 integration_limit=integration_limit)
        else:
            tester = Tester(model_path=self.model_path, debugging=debugging, dataset_type=dataset_type,
                            pretrained_weights_dir=pretrained_weights_dir,
                            image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                            rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                            h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                            h_flip_rot_270=h_flip_rot_270,
                            integration_limit=integration_limit)

        tester.evaluate(dataset=dataset)

    def display_predictions(self, dataset: Dataset, dataset_type: _DATASET_TYPES, pretrained_weights_dir: str = None,
                            debugging: bool = False, self_ensembling: bool = False,
                            image_size: int = 256, mask_size: int = 1024,
                            rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
                            h_flip: bool = False, h_flip_rot_90: bool = False,
                            h_flip_rot_180: bool = False, h_flip_rot_270: bool = False) -> None:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        if self.use_patches:
            tester = PatchTester(model_path=self.model_path, dataset_type=dataset_type,
                                 pretrained_weights_dir=pretrained_weights_dir, debugging=debugging,
                                 image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                                 rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                                 h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                                 h_flip_rot_270=h_flip_rot_270)
        else:
            tester = Tester(model_path=self.model_path, dataset_type=dataset_type,
                            pretrained_weights_dir=pretrained_weights_dir, debugging=debugging,
                            image_size=image_size, mask_size=mask_size, use_self_ensembling=self_ensembling,
                            rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                            h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                            h_flip_rot_270=h_flip_rot_270)

        tester.display_predictions(dataset=dataset)

    def predict(self, image_path: str, display_prediction: bool, dataset_type: _DATASET_TYPES,
                pretrained_weights_dir: str = None, debugging: bool = False,
                mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225),
                self_ensembling: bool = False, image_size: int = 256,
                rot_90: bool = False, rot_180: bool = False, rot_270: bool = False,
                h_flip: bool = False, h_flip_rot_90: bool = False,
                h_flip_rot_180: bool = False, h_flip_rot_270: bool = False) -> Tuple[np.array, np.array]:
        if not self.trained:
            raise Exception("Model not trained.")
        if not self.valid_model:
            raise Exception("Invalid model.")

        if self.use_patches:
            tester = PatchTester(model_path=self.model_path, dataset_type=dataset_type,
                                 pretrained_weights_dir=pretrained_weights_dir, debugging=debugging,
                                 image_size=image_size, use_self_ensembling=self_ensembling,
                                 rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                                 h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                                 h_flip_rot_270=h_flip_rot_270)
        else:
            tester = Tester(model_path=self.model_path, dataset_type=dataset_type,
                            pretrained_weights_dir=pretrained_weights_dir, debugging=debugging,
                            image_size=image_size, use_self_ensembling=self_ensembling,
                            rot_90=rot_90, rot_180=rot_180, rot_270=rot_270, h_flip=h_flip,
                            h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                            h_flip_rot_270=h_flip_rot_270)

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
        valid = os.path.exists(os.path.join(model_path, "match.pt"))

        return valid

    def __valid_patch_model(self, model_path: str) -> bool:
        valid = True
        valid = valid and self.__valid_model(os.path.join(model_path, "big"))
        valid = valid and self.__valid_model(os.path.join(model_path, "medium"))
        valid = valid and self.__valid_model(os.path.join(model_path, "small"))

        return valid

    # endregion
