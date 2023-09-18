import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
from abc import ABC
from collections import OrderedDict
from os.path import join
from typing import List
from PIL import Image
from torch import Tensor
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.transforms import transforms

from source.models.utils import BaseTester

"""
    Implementation of Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    Code modified from https://github.com/byungjae89/SPADE-pytorch/tree/master
    Paper: https://arxiv.org/abs/2005.02357
"""


class Tester(BaseTester, ABC):

    # region init

    def __init__(self, model_path: str, debugging: bool = False,
                 image_size: int = 256, mask_size: int = 1024, use_self_ensembling: bool = False,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 integration_limit: float = 0.3, top_k: int = 5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(model_path=model_path, debugging=debugging, image_size=image_size, mask_size=mask_size,
                         use_self_ensembling=use_self_ensembling, rot_90=rot_90, rot_180=rot_180, rot_270=rot_270,
                         h_flip=h_flip, h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                         h_flip_rot_270=h_flip_rot_270, integration_limit=integration_limit)
        self.top_k = top_k

        self.__load_model()
        self.__register_hooks()

    def __register_hooks(self):
        self.outputs = []

        def hook(module, input, output):
            self.outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.model.avgpool.register_forward_hook(hook)

    def __load_model(self):
        l1 = torch.from_numpy(np.load(join(self.model_path, "layer_1.npy"))).to(self.device)
        l2 = torch.from_numpy(np.load(join(self.model_path, "layer_2.npy"))).to(self.device)
        l3 = torch.from_numpy(np.load(join(self.model_path, "layer_3.npy"))).to(self.device)
        pool = torch.from_numpy(np.load(join(self.model_path, "avgpool.npy"))).to(self.device)

        self.train_outputs = [l1, l2, l3, pool]

        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
        self.model.to(self.device)
        self.model.eval()

    # endregion

    # region implement abstract methods

    def score(self, img_input) -> np.array:
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_outputs_list = [[], [], [], []]

        with torch.no_grad():
            self.model(img_input.to(self.device))

        for k, v in zip(test_outputs.keys(), self.outputs):
            test_outputs[k].append(v)
        test_outputs_list = [lst + [self.outputs[i]] for i, lst in enumerate(test_outputs_list)]
        self.outputs.clear()

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        test_outputs_list = [torch.stack(lst, dim=0) for lst in test_outputs_list]

        dist_matrix = self.__calc_dist_matrix(torch.flatten(test_outputs_list[3], 1),
                                              torch.flatten(self.train_outputs[3], 1))

        # select K nearest neighbor and take average
        topk_values, topk_indexes = torch.topk(dist_matrix, k=self.top_k, dim=1, largest=False)
        # scores = torch.mean(topk_values, 1).cpu().detach().numpy()

        score_maps = []
        # for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer
        for i in range(3):

            # construct a gallery of features at all pixel locations of the K nearest neighbors
            topk_feat_map = self.train_outputs[i][topk_indexes[0]]
            test_feat_map = test_outputs_list[i][0]
            # original:
            """feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)"""

            # modified:
            # source: https://github.com/byungjae89/SPADE-pytorch/pull/18
            # adjust dimensions to measure distance in the channel dimension for all combinations
            feat_gallery = topk_feat_map.transpose(1, 2).transpose(2, 3)  # (K, C, H, W) -> (K, H, W, C)
            feat_gallery = feat_gallery.flatten(0, 2)  # (K, H, W, C) -> (KHW, C)
            feat_gallery = feat_gallery.unsqueeze(1).unsqueeze(1)  # (KHW, C) -> (KHW, 1, 1, C)
            test_feat_map = test_feat_map.transpose(1, 2).transpose(2, 3)  # (K, C, H, W) -> (K, H, W, C)

            # calculate distance matrix
            dist_matrix_list = []
            # original:
            """for d_idx in range(feat_gallery.shape[0] // 100):"""
            # modified:
            for d_idx in range(feat_gallery.shape[0] // 100 + 1):
                dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                dist_matrix_list.append(dist_matrix.cpu())
            dist_matrix = torch.cat(dist_matrix_list, 0)

            # k nearest features from the gallery (k=1)
            score_map = torch.min(dist_matrix, dim=0)[0]

            score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=self.mask_size,
                                      mode='bilinear', align_corners=False)
            score_maps.append(score_map)

        # average distance between the features
        score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

        # apply gaussian smoothing on the score map
        # score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
        score_map = score_map.squeeze().cpu().detach().numpy()

        return score_map

    def score_with_augmentation(self, img_input):
        score_list = self._get_self_ensembling_scores(img_input)
        final_score = self._combine_scores(score_list)

        return final_score

    def preprocess_img(self, image_path: str, mean: List[float], std: List[float]) -> Tensor:
        original = Image.open(image_path).convert('RGB')

        normalize = transforms.Normalize(mean=mean, std=std)
        resize = torchvision.transforms.Resize(size=self.image_size, interpolation=TF.InterpolationMode.BILINEAR)

        transform = transforms.Compose([transforms.ToTensor(), normalize, resize])
        preprocessed = transform(original)

        preprocessed = preprocessed[None, :]

        return preprocessed

    # endregion

    # region private methods

    def __calc_dist_matrix(self, x, y):
        """Calculate Euclidean distance matrix with torch.tensor"""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))

        return dist_matrix

    # endregion
