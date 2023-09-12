import random

from source.datasets.anomaly_creator.dfc_anomaly_creator import AnomalyCreator as DFCAnomalyCreator
from source.datasets.anomaly_creator.ssaps_anomaly_creator import PatchAnomalyCreator
from source.datasets.anomaly_creator.cutpaste_anomaly_creator import CutPaste

"""
    Options to create anomalies:
    
    DFC: Deep Feature Correspondence 
        paper: (https://www.sciencedirect.com/science/article/abs/pii/S0031320322003557)
    SSAPS: Self-Supervised Augmented Patches Segmentation for Anomaly Detection 
        paper: (https://openaccess.thecvf.com/content/ACCV2022/html/Long_Self-Supervised_Augmented_Patches_Segmentation_for_Anomaly_Detection_ACCV_2022_paper.html)
    CutPaste: Self-Supervised Learning for Anomaly Detection and Localization
        paper: https://arxiv.org/abs/2104.04015
"""


class AnomalyCreator(object):

    def __init__(self, img_size, mask_size, mean, std, imagenet_dir,
                 method='dfc', dfc_anomaly_size='big', cutpaste_mode='all'):
        self.img_size = img_size
        self.mask_size = mask_size
        self.mean = mean
        self.std = std
        self.anomaly_size = dfc_anomaly_size
        self.method = method

        self.creator = DFCAnomalyCreator(img_size, mask_size, mean, std, imagenet_dir, 'all')

        if method == 'dfc':
            self.creator = DFCAnomalyCreator(img_size, mask_size, mean, std, imagenet_dir, dfc_anomaly_size)
        elif method == 'ssaps':
            self.creator = PatchAnomalyCreator()
        elif method == 'cutpaste':
            self.creator = CutPaste(mode=cutpaste_mode)
        elif method == 'all':
            self.cutpaste_creator = CutPaste(mode='all')
            self.dfc_creator = DFCAnomalyCreator(img_size, mask_size, mean, std, imagenet_dir, dfc_anomaly_size)
            self.ssaps_creator = PatchAnomalyCreator()
        else:
            print("Unknown anomaly creation method.")

    def __call__(self, img):
        if self.method == 'all':
            creator = random.choice([self.cutpaste_creator, self.dfc_creator, self.ssaps_creator])
        else:
            creator = self.creator

        img_normal, img_abnormal, mask_normal, mask_abnormal = creator(img)

        return img_normal, img_abnormal, mask_normal, mask_abnormal
