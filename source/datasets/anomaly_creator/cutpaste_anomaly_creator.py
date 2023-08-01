import random
import numpy as np
from PIL import Image
from torchvision import transforms

"""
    Implementation based on https://github.com/LilitYolyan/CutPaste
    Paper: https://arxiv.org/abs/2104.04015
"""


class CutPaste(object):

    def __init__(self, mode="all"):
        self.mode = mode

    @staticmethod
    def crop_and_paste_patch(image, patch_w, patch_h, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """

        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top), mask=mask)

        label_img = Image.new("RGB", image.size, (255, 255, 255))
        label_img.paste(patch, (paste_left, paste_top), mask=mask)

        label_mask = np.all(np.array(label_img) == [255, 255, 255], axis=-1)
        label = np.where(label_mask, 0, 1)

        return aug_image, label

    def cutpaste(self, image, area_ratio=(0.02, 0.15), aspect_ratio=((0.3, 1), (1, 3.3))):
        '''
        CutPaste augmentation

        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio

        :return: PIL image after CutPaste transformation
        '''

        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w = int(np.sqrt(patch_area * patch_aspect))
        patch_h = int(np.sqrt(patch_area / patch_aspect))
        cutpaste, mask = self.crop_and_paste_patch(image, patch_w, patch_h, rotation=False)

        return cutpaste, mask

    def cutpaste_scar(self, image, width=[2, 16], length=[10, 25], rotation=(-45, 45)):
        '''

        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation

        :return: PIL image after CutPaste-Scare transformation
        '''
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar, mask = self.crop_and_paste_patch(image, patch_w, patch_h, rotation=rotation)

        return cutpaste_scar, mask

    def __call__(self, image):
        '''
        :image: [PIL] - original image
        :return: returns original image and randomly chosen transformation
        '''

        if self.mode == "all":
            aug = random.choice([self.cutpaste, self.cutpaste_scar])
            abnormal_img, label = aug(image)
        elif self.mode == "scar":
            abnormal_img, label = self.cutpaste_scar(image)
        elif self.mode == "patch":
            abnormal_img, label = self.cutpaste(image)
        else:
            print("Unknown cutpaste mode")
            return

        mask_normal = np.zeros(image.size, dtype=np.float32)

        return np.array(image), np.array(abnormal_img), mask_normal, label
