import os
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from typing import List
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms
from typing_extensions import Literal

from source.models.DFC.backbone.vgg19 import VGG19
from source.models.DFC.backbone.vgg19_s import VGG19_S
from source.models.utils import BaseTester


class Tester(BaseTester):

    # region init

    _DATASET_TYPES = Literal["textures", "objects"]
    _CNN_LAYERS_TEXTURES = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
    _CNN_LAYERS_OBJECTS = ("relu4_3", "relu4_4", "relu5_1", "relu5_2")

    def __init__(self, model_path: str, dataset_type: _DATASET_TYPES, pretrained_weights_dir: str = None,
                 debugging: bool = False,
                 image_size: int = 256, mask_size: int = 1024, use_self_ensembling: bool = False,
                 rot_90: bool = False, rot_180: bool = False, rot_270: bool = False, h_flip: bool = False,
                 h_flip_rot_90: bool = False, h_flip_rot_180: bool = False, h_flip_rot_270: bool = False,
                 integration_limit: float = 0.3):
        super().__init__(model_path=model_path, debugging=debugging, image_size=image_size, mask_size=mask_size,
                         use_self_ensembling=use_self_ensembling, rot_90=rot_90, rot_180=rot_180, rot_270=rot_270,
                         h_flip=h_flip, h_flip_rot_90=h_flip_rot_90, h_flip_rot_180=h_flip_rot_180,
                         h_flip_rot_270=h_flip_rot_270, integration_limit=integration_limit)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset_type = dataset_type
        self.pretrained_weights_dir = pretrained_weights_dir

        self.__init_feat_layers()
        self.__load_model()

    def __init_feat_layers(self) -> None:
        if self.dataset_type == "objects":
            self.feat_layers = self._CNN_LAYERS_OBJECTS
            self._log_message("Using object layers.")
        elif self.dataset_type == "textures":
            self.feat_layers = self._CNN_LAYERS_TEXTURES
            self._log_message("Using texture layers.")
        else:
            raise Exception("Unknown dataset type. Valid types: ['textures', 'objects']")

    def __load_model(self) -> None:
        # pretrained feature extraction net
        self.feature_extraction = VGG19(pretrain=True, gradient=False, pool='avg',
                                        pretrained_weights_dir=self.pretrained_weights_dir).to(self.device)
        # trained feature estimation net
        self.feature_matching = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        self.feature_matching.load_state_dict(torch.load(os.path.join(self.model_path, 'match.pt'),
                                                         map_location=torch.device(self.device)))

        self.feature_extraction.eval()
        self.feature_matching.eval()

    # endregion

    # region implement abstract methods

    def score(self, img_input) -> np.array:  # returns score with shape (1024, 1024)
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

    def score_with_augmentation(self, img_input) -> np.array:
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
