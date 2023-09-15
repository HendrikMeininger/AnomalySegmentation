from collections import OrderedDict
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import source.evaluation.eval as evaluation
from source.datasets.test_dataset import TestDataset
from source.utils import visualization

"""
    Implementation of Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    Code modified from https://github.com/byungjae89/SPADE-pytorch/tree/master
    Paper: https://arxiv.org/abs/2005.02357
"""


class Tester(object):

    # region init

    def __init__(self, dataset_dir: str, model_path: str, debugging: bool = False, augmentation: bool = False,
                 image_size: int = 1024, mask_size: int = 1024, top_k: int = 5,
                 v_flip=False, h_flip=False, rotate=False, rotation_degrees=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_path = model_path
        self.debugging = debugging
        self.augmentation = augmentation
        self.image_size = image_size
        self.top_k = top_k
        self.mask_size = mask_size
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.rotate = rotate
        self.rotation_degrees = rotation_degrees

        self.dataset = TestDataset(path_to_dataset=dataset_dir, image_size=image_size, mask_size=1024)
        self.__load_model()

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

    # region public

    def evaluate(self) -> None:
        scores, masks = self.__predict_images()
        evaluation.print_metrics(scores, masks)

    def display_predictions(self) -> None:
        test_loader = DataLoader(dataset=self.dataset, batch_size=1)

        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.model.avgpool.register_forward_hook(hook)

        for original, preprocessed, mask in test_loader:
            if self.augmentation:
                score = self.__score_with_augmentation(preprocessed, outputs)
            else:
                score = self.__score(preprocessed, outputs)

            binary_score = evaluation.get_binary_score(score)

            original = original.squeeze()
            mask = mask.squeeze()

            visualization.display_images(img_list=[original, mask, score, binary_score],
                                         titles=['original', 'ground_truth', 'score', 'binary_score'],
                                         cols=3)

    # endregion

    # region private methods

    def __predict_images(self):
        test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)

        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.model.avgpool.register_forward_hook(hook)

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
                score = self.__score_with_augmentation(img, outputs)
            else:
                score = self.__score(img, outputs)

            scores.append(score)
            masks.append(mask.squeeze().numpy())

        return scores, masks

    def __score(self, img, outputs):
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_outputs_list = [[], [], [], []]

        with torch.no_grad():
            self.model(img.to(self.device))

        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v)
        test_outputs_list = [lst + [outputs[i]] for i, lst in enumerate(test_outputs_list)]
        outputs.clear()

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        test_outputs_list = [torch.stack(lst, dim=0) for lst in test_outputs_list]

        dist_matrix = self.calc_dist_matrix(torch.flatten(test_outputs_list[3], 1),
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

    def __score_with_augmentation(self, img_input, outputs):
        score = self.__score(img_input, outputs)
        score_list = [score]

        score_list += self.__get_flip_scores(img_input, outputs)
        if self.rotate:
            score_list += self.__get_rotation_scores(img_input, outputs)

        final_score = self.__combine_scores(score_list)
        return final_score

    def __get_flip_scores(self, img_input, outputs):
        scores = []

        if self.v_flip:
            # Flip the image vertically
            vertical_flip = torch.flip(img_input, dims=[2])
            vertical_flip_score = self.__score(vertical_flip, outputs)
            vertical_flip_score = np.flipud(vertical_flip_score)
            scores.append(vertical_flip_score)

        if self.h_flip:
            # Flip the image horizontally
            horizontal_flip = torch.flip(img_input, dims=[3])
            horizontal_flip_score = self.__score(horizontal_flip, outputs)
            horizontal_flip_score = np.fliplr(horizontal_flip_score)
            scores.append(horizontal_flip_score)

        return scores

    def __get_rotation_scores(self, img_input, outputs):
        res = []

        if 90 in self.rotation_degrees:
            # Rotate image 90
            rotated_90 = TF.rotate(img_input, -90)
            rotated_90_score = self.__score(rotated_90, outputs)
            rotated_90_score = np.rot90(rotated_90_score)
            res.append(rotated_90_score)

        if 180 in self.rotation_degrees:
            # Rotate image 180
            rotated_180 = TF.rotate(img_input, -180)
            rotated_180_score = self.__score(rotated_180, outputs)
            rotated_180_score = np.rot90(rotated_180_score, k=2)
            res.append(rotated_180_score)

        if 270 in self.rotation_degrees:
            # Rotate image 270
            rotated_270 = TF.rotate(img_input, -270)
            rotated_270_score = self.__score(rotated_270, outputs)
            rotated_270_score = np.rot90(rotated_270_score, k=3)
            res.append(rotated_270_score)

        return res

    def __combine_scores(self, score_list):
        score_list = np.mean(score_list, axis=0)

        return score_list

    def calc_dist_matrix(self, x, y):
        """Calculate Euclidean distance matrix with torch.tensor"""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
        return dist_matrix

    # endregion
