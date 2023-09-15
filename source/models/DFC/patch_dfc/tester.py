import torch
import torch.nn.functional as F
import os
import numpy as np
import torchvision.transforms.functional as TF
import torchvision
from typing import List, Tuple

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, transforms, Resize
from typing_extensions import Literal

import source.evaluation.eval as evaluation
from source.models.DFC.backbone.vgg19 import VGG19
from source.models.DFC.backbone.vgg19_s import VGG19_S
from source.utils import visualization
from source.models.utils import BaseTester


class Tester(BaseTester):

    # region init

    _DATASET_TYPES = Literal["textures", "objects"]
    _CNN_LAYERS_TEXTURES = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
    _CNN_LAYERS_OBJECTS = ("relu4_3", "relu4_4", "relu5_1", "relu5_2")

    def __init__(self, model_path: str, dataset_type: _DATASET_TYPES, pretrained_weights_dir: str = None,
                 debugging: bool = False, patch_size: int = 256,
                 image_size: int = 1024, mask_size: int = 1024, use_self_ensembling: bool = False,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 integration_limit: float = 0.3):
        self.patch_size = patch_size
        self.resize_transform = Resize(size=patch_size, interpolation=TF.InterpolationMode.BILINEAR)

        super().__init__(model_path=model_path, debugging=debugging, image_size=image_size, mask_size=mask_size,
                         use_self_ensembling=use_self_ensembling, rot_90=rot_90, rot_180=rot_180, rot_270=rot_270,
                         h_flip=h_flip, h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                         h_flip_rot_270=h_flip_rot_270, integration_limit=integration_limit)

        self.dataset_type = dataset_type
        self.pretrained_weights_dir = pretrained_weights_dir

        self.__init_feat_layers()
        self.__load_models()

    def __load_models(self) -> None:
        big_model_path = os.path.join(self.model_path, 'big/match.pt')
        medium_model_path = os.path.join(self.model_path, 'medium/match.pt')
        small_model_path = os.path.join(self.model_path, 'small/match.pt')

        # pretrained feature extraction net
        self.feature_extraction = VGG19(pretrain=True, gradient=False, pool='avg',
                                        pretrained_weights_dir=self.pretrained_weights_dir).to(self.device)
        self.feature_extraction.eval()

        # trained feature estimation net
        self.feature_matching_big = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching_big.load_state_dict(torch.load(big_model_path,
                                                             map_location=torch.device(self.device)))
        self.feature_matching_big.eval()

        self.feature_matching_medium = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching_medium.load_state_dict(torch.load(medium_model_path,
                                                                map_location=torch.device(self.device)))
        self.feature_matching_medium.eval()

        self.feature_matching_small = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching_small.load_state_dict(torch.load(small_model_path,
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

    # region implement abstract methods

    def score_with_augmentation(self, img_input):
        score_list = self._get_self_ensembling_scores(img_input)
        final_score = self._combine_scores(score_list)

        return final_score

    def score(self, img_input) -> np.array:
        big_patches_score = self.__score_big_patches(img_input, 0)
        medium_patches_score = self.__score_medium_patches(img_input, 0)
        small_patches_score = self.__score_small_patches(img_input, 0)

        final_score = self.__calc_final_score(big_patches_score, medium_patches_score, small_patches_score)

        return final_score

    def preprocess_img(self, image_path: str, mean: List[float], std: List[float]):
        original = Image.open(image_path).convert('RGB')

        normalize = transforms.Normalize(mean=mean, std=std)
        resize = torchvision.transforms.Resize(size=self.image_size, interpolation=TF.InterpolationMode.BILINEAR)

        transform = transforms.Compose([transforms.ToTensor(), normalize, resize])
        preprocessed = transform(original)

        preprocessed = preprocessed[None, :]

        return preprocessed

    # endregion

    # region private methods

    # maximum of big, medium, small at each pixel
    def __calc_final_score(self, big, medium, small):
        # score = np.maximum(big, np.maximum(medium, small))
        score = (big + medium + small) / 3
        return score

    def __score_big_patches(self, img, threshold):
        return self.__score_patch(img, self.feature_matching_big, threshold, 1024)

    def __score_medium_patches(self, img, threshold):
        score = np.zeros(shape=(self.mask_size, self.mask_size), dtype=float)

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
            anomaly_map += F.interpolate(
                torch.pow(surrogate_label[feat_layer] - prediction[feat_layer], 2).mean(dim=1, keepdim=True),
                size=(out_size, out_size), mode="bilinear", align_corners=True)

        score = anomaly_map.data.cpu().numpy().squeeze()
        score[score < threshold] = 0

        return score

    # endregion
