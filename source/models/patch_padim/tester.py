import math
from os.path import join
from typing import Tuple, List
import gc

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from source.datasets.test_dataset import TestDataset
from source.utils import visualization
from source.utils.performance_measurement import Timer
from source.models.padim_backbone.padim import PaDiM
from source.models.padim_backbone.utils import mean_smoothing
import source.evaluation.eval as evaluation


class Tester(object):

    # region init

    def __init__(self, dataset, model_dir: str, debugging: bool = False, augmentation: bool = False,
                 image_size: int = 256, mask_size: int = 1024, backbone="resnet18"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset = dataset
        self.model_dir = model_dir
        self.debugging = debugging
        self.augmentation = augmentation
        self.image_size = image_size
        self.mask_size = mask_size
        self.backbone = backbone

        self.batch_size = 1
        self.integration_limit = 0.3

        self.resize_transform = \
            torchvision.transforms.Resize(size=256, interpolation=TF.InterpolationMode.BILINEAR)

        self.__load_models()

    def __load_models(self):
        self.padim_big = self.__load_model(join(self.model_dir, "big"))
        self.padim_medium = self.__load_model(join(self.model_dir, "medium"))
        self.padim_small = self.__load_model(join(self.model_dir, "small"))

    def __load_model(self, model_path):
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
                                     img_size=256)
        return padim

    @classmethod
    def from_dataset_dir(cls, dataset_dir: str, model_dir: str, debugging: bool = False, augmentation: bool = False,
                         image_size: int = 1024, mask_size: int = 1024, backbone="resnet18"):
        test_dataset = TestDataset(path_to_dataset=dataset_dir, image_size=image_size, mask_size=mask_size)
        return cls(dataset=test_dataset, model_dir=model_dir, debugging=debugging, augmentation=augmentation,
                   image_size=image_size, backbone=backbone, mask_size=mask_size)

    # endregion

    # region public methods

    def evaluate(self):
        scores, masks = self.__predict_images()
        del self.padim_big
        del self.padim_medium
        del self.padim_small
        del self.dataset
        gc.collect()

        evaluation.print_metrics(scores=scores, masks=masks)

    def display_predictions(self):
        test_loader = DataLoader(dataset=self.dataset, batch_size=1)

        for original, preprocessed, mask in test_loader:
            if self.augmentation:
                score = self.__score_with_augmentation(preprocessed)
            else:
                score = self.__score(preprocessed)

            original = original.squeeze()
            mask = mask.squeeze()
            binary_score = evaluation.get_binary_score(score)
            visualization.display_images(img_list=[original, mask, score, binary_score],
                                         titles=['original', 'ground_truth', 'score', 'binary_score'],
                                         cols=3)

    # end region

    # region private methods

    def __predict_images(self) -> Tuple[List[np.array], List[np.array]]:
        test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)

        scores = []
        masks = []

        number_of_paths = len(self.dataset)
        if self.debugging:
            print("Testing ", number_of_paths, " images.")
        count = 1

        for _, img, mask in test_dataloader:
            if count % 10 == 0 and self.debugging:
                print("Predicting img {}/{}".format(count, number_of_paths))
            count += 1

            if self.augmentation:
                score = self.__score_with_augmentation(img)
            else:
                score = self.__score(img)

            scores.append(score)
            masks.append(mask.squeeze().numpy())

        return scores, masks

    def __score(self, img_input) -> np.array:
        big_patches_score = self.__score_big_patches(img_input)
        medium_patches_score = self.__score_medium_patches(img_input)
        small_patches_score = self.__score_small_patches(img_input)

        final_score = self.__calc_final_score(big_patches_score, medium_patches_score, small_patches_score)

        return final_score

    def __calc_final_score(self, big, medium, small):
        # score = np.max(big, np.max(medium, small))
        score = (big + medium + small) / 3
        return score

    def __score_big_patches(self, img):
        return self.__score_patch(img, self.padim_big, 1024)

    def __score_medium_patches(self, img):
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

    def __score_small_patches(self, img):
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

    def __score_patch(self, patch, model, out_size):
        img = self.resize_transform(patch).to(self.device)

        distances = model.predict(img)
        w = int(math.sqrt(distances.numel()))
        raw_score = distances.reshape(1, 1, w, w)
        raw_score = F.interpolate(raw_score, size=(out_size, out_size), mode="bilinear",
                                  align_corners=True)

        raw_score = raw_score.detach().cpu().numpy().squeeze()

        return raw_score

    def __score_with_augmentation(self, img_input):
        score = self.__score(img_input)
        score_list = [score]

        flip_score_list = self.__get_flip_scores(img_input)
        score_list = score_list + flip_score_list

        rotation_score_list = self.__get_rotation_scores(img_input)
        score_list = score_list + rotation_score_list

        final_score = self.__combine_scores(score_list)
        return final_score

    def __get_flip_scores(self, img_input):
        # Flip the image vertically
        vertical_flip = torch.flip(img_input, dims=[2])
        vertical_flip_score = self.__score(vertical_flip)
        vertical_flip_score = np.flipud(vertical_flip_score)

        # Flip the image horizontally
        horizontal_flip = torch.flip(img_input, dims=[3])
        horizontal_flip_score = self.__score(horizontal_flip)
        horizontal_flip_score = np.fliplr(horizontal_flip_score)

        # Flip the image horizontally and vertically
        double_flip = torch.flip(vertical_flip, dims=[3])
        double_flip_score = self.__score(double_flip)
        double_flip_score = np.flipud(np.fliplr(double_flip_score))

        scores = [vertical_flip_score, horizontal_flip_score, double_flip_score]

        return scores

    def __get_rotation_scores(self, img_input):
        # Rotate image 90
        rotated_90 = TF.rotate(img_input, -90)
        rotated_90_score = self.__score(rotated_90)
        rotated_90_score = np.rot90(rotated_90_score)

        # Rotate image 180
        rotated_180 = TF.rotate(img_input, -180)
        rotated_180_score = self.__score(rotated_180)
        rotated_180_score = np.rot90(rotated_180_score, k=2)

        # Rotate image 270
        rotated_270 = TF.rotate(img_input, -270)
        rotated_270_score = self.__score(rotated_270)
        rotated_270_score = np.rot90(rotated_270_score, k=3)

        scores = [rotated_90_score, rotated_180_score, rotated_270_score]

        return scores

    def __combine_scores(self, score_list):
        score_list = np.mean(score_list, axis=0)

        return score_list

    # end region
