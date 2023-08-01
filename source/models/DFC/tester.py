import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple

import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ColorJitter, RandomAdjustSharpness, GaussianBlur, RandomAutocontrast

import source.evaluation.eval as evaluation
from source.datasets.train_dataset import get_train_img_paths, TrainDataset
from source.datasets.test_dataset import TestDataset
from source.models.DFC_backbone.vgg19 import VGG19
from source.models.DFC_backbone.vgg19_s import VGG19_S
from source.utils import visualization
import torchvision.transforms.functional as TF


class Tester(object):

    # region init

    def __init__(self, dataset, dataset_type: str, model_path: str, pretrained_weights_dir: str = None,
                 debugging: bool = False, augmentation: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.model_path = model_path
        self.pretrained_weights_dir = pretrained_weights_dir
        self.debugging = debugging
        self.augmentation = augmentation

        self.image_size = 1024
        self.mask_size = 1024
        # TODO support other batch sizes for speedup
        self.batch_size = 1
        self.integration_limit = 0.3

        self.__init_feat_layers()
        self.__load_model()

    @classmethod
    def from_dataset_dir(cls, dataset_dir: str, dataset_type: str, model_path: str, pretrained_weights_dir: str = None,
                         debugging: bool = False, augmentation: bool = False):
        test_dataset = TestDataset(path_to_dataset=dataset_dir, mask_size=1024, image_size=256)
        return cls(dataset=test_dataset, dataset_type=dataset_type, model_path=model_path,
                   pretrained_weights_dir=pretrained_weights_dir, debugging=debugging, augmentation=augmentation)

    def __load_model(self) -> None:
        # pretrained feature extraction net
        self.feature_extraction = VGG19(pretrain=True, gradient=False, pool='avg',
                                        pretrained_weights_dir=self.pretrained_weights_dir).to(self.device)
        # trained feature estimation net
        self.feature_matching = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching.load_state_dict(torch.load(self.model_path,
                                                         map_location=torch.device(self.device)))

        self.feature_extraction.eval()
        self.feature_matching.eval()

    def __init_feat_layers(self):
        cnn_layers_textures = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
        cnn_layers_objects = ("relu4_3", "relu4_4", "relu5_1", "relu5_2")
        if self.dataset_type == 'objects':
            self.feat_layers = cnn_layers_objects
        elif self.dataset_type == 'textures':
            self.feat_layers = cnn_layers_textures
        else:
            self.feat_layers = ("relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4")
            print("Using other layers.")

    # endregion

    # region public methods

    def evaluate(self) -> None:
        scores, masks = self.__predict_images()
        evaluation.print_metrics(scores, masks)

    def display_predictions(self) -> None:
        test_loader = DataLoader(dataset=self.dataset, batch_size=1)

        for original, preprocessed, mask in test_loader:
            if self.augmentation:
                score = self.__score_with_augmentation(preprocessed)
            else:
                score = self.__score(preprocessed)

            binary_score = evaluation.get_binary_score(score)

            original = original.squeeze()
            mask = mask.squeeze()

            visualization.display_images(img_list=[original, mask, score, binary_score],
                                         titles=['original', 'ground_truth', 'score', 'binary_score'],
                                         cols=3)

    def get_metrics(self):
        scores, masks = self.__predict_images()
        binary_scores = evaluation.calculate_binary_scores(scores)

        image_level_roc, pixel_level_roc = evaluation.calculate_au_roc(ground_truth=masks,
                                                                       predictions=scores)
        au_pro, pro_curve = evaluation.calculate_au_pro(ground_truth=masks, predictions=scores,
                                                        integration_limit=self.integration_limit)
        avg_iou = evaluation.calculate_avg_iou(ground_truth=masks, binary_scores=binary_scores)

        return pixel_level_roc, au_pro, avg_iou

    # endregion

    # region prediction

    def __predict_images(self) -> Tuple[List[np.array], List[np.array]]:
        test_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

        number_of_paths = len(self.dataset)
        print("Testing ", number_of_paths, " images.")
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
                score = self.__score(preprocessed)

            scores.append(score)
            masks.append(mask.squeeze().numpy())

        return scores, masks

    def __score(self, img_input) -> np.array:  # returns score with shape (1024, 1024)
        img = img_input.to(self.device)
        with torch.no_grad():
            surrogate_label = self.feature_extraction(img, self.feat_layers)
            prediction = self.feature_matching(img, self.feat_layers)

        anomaly_map = 0
        for feat_layer in self.feat_layers:
            anomaly_map += F.interpolate(
                torch.pow(surrogate_label[feat_layer] - prediction[feat_layer], 2).mean(dim=1, keepdim=True),
                size=(self.mask_size, self.mask_size), mode="bilinear", align_corners=True)

        scores = anomaly_map.data.cpu().numpy().squeeze()

        return scores

    def __score_with_augmentation(self, img_input):
        score = self.__score(img_input)
        score_list = [score]

        score_list = score_list + self.__get_flip_scores(img_input)
        # score_list = score_list + self.__get_rotation_scores(img_input)
        # score_list = score_list + self.__get_brightness_scores(img_input, threshold)

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

        # return [vertical_flip_score, horizontal_flip_score, double_flip_score]
        return [vertical_flip_score]

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

        return [rotated_90_score, rotated_180_score, rotated_270_score]

    def __get_brightness_scores(self, img_input):
        blurred_img = GaussianBlur(kernel_size=3)(img_input)
        blurred_score = self.__score(blurred_img)

        random_sharpness = RandomAdjustSharpness(sharpness_factor=2, p=1)
        sharpness_img = random_sharpness(img_input)
        sharpness_score = self.__score(sharpness_img)

        random_contrast = RandomAutocontrast(p=1)
        contrast_img = random_contrast(img_input)
        contrast_score = self.__score(contrast_img)

        random_color_jitter = transforms.ColorJitter(brightness=(0.5, 1.5),
                                                     contrast=1,
                                                     saturation=(0.5, 1.5),
                                                     hue=(-0.1, 0.1))
        random_color_img = random_color_jitter(img_input)
        random_color_score = self.__score(random_color_img)

        return [blurred_score, sharpness_score, contrast_score, random_color_score]

    def __combine_scores(self, score_list):
        res = np.mean(score_list, axis=0)

        return res

    # endregion
