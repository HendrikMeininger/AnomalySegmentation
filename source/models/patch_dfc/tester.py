import os.path

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, transforms

import source.evaluation.eval as evaluation
from source.datasets.train_dataset import get_train_img_paths, TrainDataset
from source.datasets.test_dataset import TestDataset
from source.models.DFC_backbone.vgg19 import VGG19
from source.models.DFC_backbone.vgg19_s import VGG19_S
from source.utils import visualization
import torchvision.transforms.functional as TF


class Tester(object):

    # region init

    def __init__(self, dataset, dataset_type: str, big_model_path: str, medium_model_path: str, small_model_path: str,
                 pretrained_weights_dir: str = None, debugging: bool = False, augmentation: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.big_model_path = big_model_path
        self.medium_model_path = medium_model_path
        self.small_model_path = small_model_path
        self.pretrained_weights_dir = pretrained_weights_dir
        self.debugging = debugging
        self.augmentation = augmentation

        self.image_size = 1024
        # TODO support other batch sizes for speedup
        self.batch_size = 1
        self.integration_limit = 0.3
        self.resize_transform = \
            torchvision.transforms.Resize(size=256, interpolation=InterpolationMode.BILINEAR)

        self.__init_feat_layers()
        self.__load_models()

    @classmethod
    def from_dataset_dir(cls, dataset_dir: str, dataset_type: str, big_model_path: str, medium_model_path: str,
                         small_model_path: str, pretrained_weights_dir: str = None, debugging: bool = False,
                         augmentation: bool = False):
        test_dataset = TestDataset(path_to_dataset=dataset_dir, image_size=1024, mask_size=1024)
        return cls(dataset=test_dataset, dataset_type=dataset_type, big_model_path=big_model_path,
                   medium_model_path=medium_model_path, small_model_path=small_model_path,
                   pretrained_weights_dir=pretrained_weights_dir, debugging=debugging, augmentation=augmentation)

    def __load_models(self) -> None:
        # pretrained feature extraction net
        self.feature_extraction = VGG19(pretrain=True, gradient=False, pool='avg',
                                        pretrained_weights_dir=self.pretrained_weights_dir).to(self.device)
        self.feature_extraction.eval()

        # trained feature estimation net
        self.feature_matching_big = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching_big.load_state_dict(torch.load(self.big_model_path,
                                                             map_location=torch.device(self.device)))
        self.feature_matching_big.eval()

        self.feature_matching_medium = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching_medium.load_state_dict(torch.load(self.medium_model_path,
                                                                map_location=torch.device(self.device)))
        self.feature_matching_medium.eval()

        self.feature_matching_small = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching_small.load_state_dict(torch.load(self.small_model_path,
                                                               map_location=torch.device(self.device)))
        self.feature_matching_small.eval()

    def __init_feat_layers(self):
        cnn_layers_textures = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
        cnn_layers_objects = ("relu4_3", "relu4_4", "relu5_1", "relu5_2")
        if self.dataset_type == 'objects':
            self.feat_layers = cnn_layers_objects
        elif self.dataset_type == 'textures':
            self.feat_layers = cnn_layers_textures
        else:
            print('Unknown dataset type.')

    # endregion

    # region public methods

    def evaluate(self) -> None:
        scores, masks = self.__predict_images()
        evaluation.print_metrics(scores, masks)

    def display_predictions(self) -> None:
        test_loader = DataLoader(dataset=self.dataset, batch_size=1)

        for original, preprocessed, mask in test_loader:
            score, big, medium, small = self.__score(preprocessed)

            binary_score = evaluation.get_binary_score(score)

            original = original.squeeze()
            mask = mask.squeeze()

            visualization.display_images(img_list=[original, mask, np.zeros_like(score),
                                                   score, binary_score, np.zeros_like(score),
                                                   big, medium, small],
                                         titles=['original', 'ground_truth', '',
                                                 'score', 'binary_score', '',
                                                 'big', 'medium', 'small'],
                                         cols=3)

    def get_metrics(self):
        scores, masks = self.__predict_images()
        binary_scores = evaluation.calculate_binary_scores(scores)

        au_roc, roc_curve, optimal_threshold, highest_accuracy = evaluation.calculate_au_roc(ground_truth=masks,
                                                                                             predictions=scores)
        au_pro, pro_curve = evaluation.calculate_au_pro(ground_truth=masks, predictions=scores,
                                                        integration_limit=self.integration_limit)
        avg_iou = evaluation.calculate_avg_iou(ground_truth=masks, binary_scores=binary_scores)

        return au_roc, au_pro, avg_iou

    # endregion

    # region prediction

    def __predict_images(self) -> Tuple[List[np.array], List[np.array]]:
        test_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)

        number_of_paths = len(self.dataset)
        count = 1

        scores = []
        masks = []

        for original, preprocessed, mask in test_loader:
            if count % 10 == 0 and self.debugging:
                print("Predicting img {}/{}".format(count, number_of_paths))
            count += 1

            if self.augmentation:
                score = self.__score_with_augmentation(preprocessed)
            else:
                score, big, medium, small = self.__score(preprocessed)

            scores.append(score)
            masks.append(mask.squeeze().numpy())

        return scores, masks

    def __score_with_augmentation(self, img_input):
        score, _, _, _ = self.__score(img_input)
        score_list = [score]

        score_list = score_list + self.__get_flip_scores(img_input)
        score_list = score_list + self.__get_rotation_scores(img_input)
        # score_list = score_list + self.__get_brightness_scores(img_input, threshold)

        final_score = self.__combine_scores(score_list)
        return final_score

    def __get_flip_scores(self, img_input):
        # Flip the image vertically
        vertical_flip = torch.flip(img_input, dims=[2])
        vertical_flip_score, _, _, _ = self.__score(vertical_flip)
        vertical_flip_score = np.flipud(vertical_flip_score)

        # Flip the image horizontally
        horizontal_flip = torch.flip(img_input, dims=[3])
        horizontal_flip_score, _, _, _ = self.__score(horizontal_flip)
        horizontal_flip_score = np.fliplr(horizontal_flip_score)

        # Flip the image horizontally and vertically
        double_flip = torch.flip(vertical_flip, dims=[3])
        double_flip_score, _, _, _ = self.__score(double_flip)
        double_flip_score = np.flipud(np.fliplr(double_flip_score))

        return [vertical_flip_score, horizontal_flip_score, double_flip_score]

    def __get_rotation_scores(self, img_input):
        # Rotate image 90
        rotated_90 = TF.rotate(img_input, -90)
        rotated_90_score, _, _, _ = self.__score(rotated_90)
        rotated_90_score = np.rot90(rotated_90_score)

        # Rotate image 180
        rotated_180 = TF.rotate(img_input, -180)
        rotated_180_score, _, _, _ = self.__score(rotated_180)
        rotated_180_score = np.rot90(rotated_180_score, k=2)

        # Rotate image 270
        rotated_270 = TF.rotate(img_input, -270)
        rotated_270_score, _, _, _ = self.__score(rotated_270)
        rotated_270_score = np.rot90(rotated_270_score, k=3)

        return [rotated_90_score, rotated_180_score, rotated_270_score]

    def __get_brightness_scores(self, img_input):
        color_img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img_input)
        color_score, _, _, _ = self.__score(color_img)

        sharpness_img = transforms.RandomAdjustSharpness(sharpness_factor=2, p=1)(img_input)
        sharpness_score, _, _, _ = self.__score(sharpness_img)

        """auto_contrast_img = transforms.RandomAutocontrast(p=1)(img_input)
        auto_contrast_score, _, _, _ = self.__score(auto_contrast_img, threshold)"""

        blurred_img = transforms.GaussianBlur(kernel_size=3)(img_input)
        blurred_score, _, _, _ = self.__score(blurred_img)

        return [color_score, sharpness_score, blurred_score]

    def __combine_scores(self, score_list):
        res = np.mean(score_list, axis=0)

        return res

    def __score(self, img_input) -> np.array:
        big_patches_score = self.__score_big_patches(img_input, 0)
        medium_patches_score = self.__score_medium_patches(img_input, 0)
        small_patches_score = self.__score_small_patches(img_input, 0)

        final_score = self.__calc_final_score(big_patches_score, medium_patches_score, small_patches_score)

        return final_score, big_patches_score, medium_patches_score, small_patches_score

    # maximum of big, medium, small at each pixel
    def __calc_final_score(self, big, medium, small):
        # score = np.maximum(big, np.maximum(medium, small))
        score = (big + medium + small) / 3
        return score

    def __score_big_patches(self, img, threshold):
        return self.__score_patch(img, self.feature_matching_big, threshold, 1024)

    def __score_medium_patches(self, img, threshold):
        score = np.zeros(shape=(self.image_size, self.image_size), dtype=float)

        width, height = 512, 512

        for x in range(0, 1024, width):
            for y in range(0, 1024, height):
                patch = img[:, :, x:x + width, y:y + height]

                patch_score = self.__score_patch(patch, self.feature_matching_medium, threshold, 512)
                score_x = x
                score_y = y
                score[score_x:score_x + width, score_y:score_y + height] = patch_score

        return score

    def __score_small_patches(self, img, threshold):
        score = np.zeros(shape=(self.image_size, self.image_size), dtype=float)

        width, height = 256, 256

        for x in range(0, 1024, width):
            for y in range(0, 1024, height):
                patch = img[:, :, x:x + width, y:y + height]

                patch_score = self.__score_patch(patch, self.feature_matching_small, threshold, 256)
                score_x = x
                score_y = y
                score[score_x:score_x + width, score_y:score_y + height] = patch_score

        return score

    def __score_patch(self, patch, model, threshold, out_size):
        img = self.resize_transform(patch).to(self.device)

        with torch.no_grad():
            surrogate_label = self.feature_extraction(img, self.feat_layers)
            prediction = model(img, self.feat_layers)

        anomaly_map = 0
        for feat_layer in self.feat_layers:
            """anomaly_map += F.interpolate(
                torch.pow(surrogate_label[feat_layer] - prediction[feat_layer], 2).mean(dim=1, keepdim=True),
                size=(out_size, out_size), mode="bilinear", align_corners=True)"""
            anomaly_map += F.interpolate(
                torch.max(torch.norm(surrogate_label[feat_layer] - prediction[feat_layer], dim=(0)).unsqueeze(0),
                          dim=1)[0].unsqueeze(0),
                size=(out_size, out_size), mode="bilinear", align_corners=True)

        score = anomaly_map.data.cpu().numpy().squeeze()
        score[score < threshold] = 0

        return score

    # endregion