import torchvision.transforms.functional as TF
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from abc import abstractmethod

import source.evaluation.eval as evaluation
from source.datasets.dataset import Dataset
from source.utils import visualization


class BaseTester(object):

    # region init

    def __init__(self, model_path: str, debugging: bool = False,
                 image_size: int = 256, mask_size: int = 1024, use_self_ensembling: bool = False,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 integration_limit: float = 0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_path = model_path
        self.debugging = debugging
        self.image_size = image_size
        self.mask_size = mask_size

        self.use_self_ensembling = use_self_ensembling
        self.rot_90 = rot_90
        self.rot_180 = rot_180
        self.rot_270 = rot_270
        self.h_flip = h_flip
        self.h_flip_rot_90 = h_flip_rot_90
        self.h_flip_rot_180 = h_flip_rot_180
        self.h_flip_rot_270 = h_flip_rot_270

        self.integration_limit = integration_limit

    # endregion

    # region public methods

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        scores, masks = self.__predict_images(dataset)

        return evaluation.get_metrics(scores=scores, masks=masks, debugging=self.debugging)

    def predict(self, image_path: str,
                mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225)):
        preprocessed = self.preprocess_img(image_path=image_path,
                                           mean=list(mean),
                                           std=list(std))

        if self.use_self_ensembling:
            score = self.score_with_augmentation(preprocessed)
        else:
            score = self.score(preprocessed)

        binary_score = evaluation.get_binary_score(score)

        return score, binary_score

    def display_predictions(self, dataset: Dataset) -> None:
        test_dataloader: DataLoader = dataset.get_test_dataloader()

        for original, preprocessed, mask in test_dataloader:
            if self.use_self_ensembling:
                score = self.score_with_augmentation(preprocessed)
            else:
                score = self.score(preprocessed)

            original = original.squeeze()
            mask = mask.squeeze()
            binary_score = evaluation.get_binary_score(score)
            visualization.display_images(img_list=[original, mask, score, binary_score],
                                         titles=['original', 'ground_truth', 'score', 'binary_score'],
                                         cols=3)

    # endregion

    # region abstract methods

    @abstractmethod
    def score(self, img_input) -> np.array:
        pass

    @abstractmethod
    def score_with_augmentation(self, img_input) -> np.array:
        pass

    @abstractmethod
    def preprocess_img(self, image_path: str, mean: List[float], std: List[float]) -> Tensor:
        pass

    # endregion

    # region private methods

    def __predict_images(self, dataset: Dataset) -> Tuple[List[np.array], List[np.array]]:
        test_dataloader: DataLoader = dataset.get_test_dataloader()

        number_of_paths = len(test_dataloader.dataset)
        self.__log_message(f"Testing {number_of_paths} images.")
        count = 1

        scores = []
        masks = []

        for _, preprocessed, mask in test_dataloader:
            if count % 10 == 0:
                self.__log_message("Predicting img {}/{}".format(count, number_of_paths))
            count += 1

            if self.use_self_ensembling:
                score = self.score_with_augmentation(preprocessed)
            else:
                score = self.score(preprocessed)

            scores.append(score)
            masks.append(mask.squeeze().numpy())

        return scores, masks

    def __get_self_ensembling_scores(self, img_input) -> List[np.array]:
        score = self.score(img_input)
        score_list = [score]

        if self.rot_90:
            rotated_90 = TF.rotate(img_input, -90)
            rotated_90_score = self.score(rotated_90)
            rotated_90_score = np.rot90(rotated_90_score)
            score_list.append(rotated_90_score)
        if self.rot_180:
            rotated_180 = TF.rotate(img_input, -180)
            rotated_180_score = self.score(rotated_180)
            rotated_180_score = np.rot90(rotated_180_score, k=2)
            score_list.append(rotated_180_score)
        if self.rot_270:
            rotated_270 = TF.rotate(img_input, -270)
            rotated_270_score = self.score(rotated_270)
            rotated_270_score = np.rot90(rotated_270_score, k=3)
            score_list.append(rotated_270_score)
        if self.h_flip:
            horizontal_flip = torch.flip(img_input, dims=[3])
            horizontal_flip_score = self.score(horizontal_flip)
            horizontal_flip_score = np.fliplr(horizontal_flip_score)
            score_list.append(horizontal_flip_score)
        if self.h_flip_rot_90:
            flipped_rotated_90 = TF.rotate(torch.flip(img_input, dims=[3]), -90)
            flipped_rotated_90_score = self.score(flipped_rotated_90)
            flipped_rotated_90_score = np.fliplr(np.rot90(flipped_rotated_90_score))
            score_list.append(flipped_rotated_90_score)
        if self.h_flip_rot_180:
            flipped_rotated_180 = TF.rotate(torch.flip(img_input, dims=[3]), -180)
            flipped_rotated_180_score = self.score(flipped_rotated_180)
            flipped_rotated_180_score = np.fliplr(np.rot90(flipped_rotated_180_score, k=2))
            score_list.append(flipped_rotated_180_score)
        if self.h_flip_rot_270:
            flipped_rotated_270 = TF.rotate(torch.flip(img_input, dims=[3]), -270)
            flipped_rotated_270_score = self.score(flipped_rotated_270)
            flipped_rotated_270_score = np.fliplr(np.rot90(flipped_rotated_270_score, k=3))
            score_list.append(flipped_rotated_270_score)

        return score_list

    def __combine_scores(self, score_list):
        res = np.mean(score_list, axis=0)

        return res

    def __log_message(self, message):
        if self.debugging:
            print(message)

    # endregion
