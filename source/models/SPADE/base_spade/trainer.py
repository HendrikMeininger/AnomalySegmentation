import os
import numpy as np
import torch
from os.path import join
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

from source.datasets.dataset import Dataset


"""
    Implementation of SPADE: Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    Code based on: https://github.com/byungjae89/SPADE-pytorch/tree/master
    Paper: https://arxiv.org/abs/2005.02357
"""


class Trainer(object):

    # region init

    def __init__(self, output_dir: str, dataset: Dataset, debugging: bool,
                 batch_size: int = 32, image_size: int = 256):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_dir = output_dir
        self.dataset = dataset
        self.debugging = debugging
        self.batch_size = batch_size
        self.image_size = image_size

        self.__init_model()

    def __init_model(self) -> None:
        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
        self.model.to(self.device)
        self.model.eval()

    # endregion

    # region public methods

    def train(self) -> None:
        train_loader = self.dataset.get_train_dataloader(self.batch_size)

        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.model.avgpool.register_forward_hook(hook)

        train_outputs = [[], [], [], []]

        for img_normal, img_abnormal, mask_normal, mask_abnormal in train_loader:
            img_normal = img_normal.to(self.device)
            # model prediction
            with torch.no_grad():
                pred = self.model(img_normal)
                del pred
                del img_normal
            # get intermediate layer outputs
            for i in range(len(train_outputs)):
                train_outputs[i].append(outputs[i])
            # train_outputs = [lst + [outputs[i]] for i, lst in enumerate(train_outputs)]

            # initialize hook outputs
            outputs = []
        for i in range(len(train_outputs)):
            train_outputs[i] = np.array(torch.cat(train_outputs[i], 0).cpu())
        # train_outputs = [np.array(torch.stack(lst, dim=0).cpu()) for lst in train_outputs]

        self.__save_model(train_outputs=train_outputs)

    # endregion

    # region private methods

    def __save_model(self, train_outputs) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        l1_path = join(self.output_dir, "layer_1.npy")
        np.save(l1_path, train_outputs[0])

        l2_path = join(self.output_dir, "layer_2.npy")
        np.save(l2_path, train_outputs[1])

        l3_path = join(self.output_dir, "layer_3.npy")
        np.save(l3_path, train_outputs[2])

        pool_path = join(self.output_dir, "avgpool.npy")
        np.save(pool_path, train_outputs[3])

    # endregion
