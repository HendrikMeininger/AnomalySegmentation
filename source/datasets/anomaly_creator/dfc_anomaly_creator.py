import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize

from source.datasets.imagenet import ILSVRC
from source.utils import visualization


class AnomalyCreator(object):

    # region init

    def __init__(self, img_size, mask_size, mean, std, imagenet_dir, anomaly_size='all'):
        self.img_size = img_size
        self.mask_size = mask_size
        self.mean = mean
        self.std = std
        self.anomaly_size = anomaly_size

        self.__load_imagenet(imagenet_dir)
        if self.anomaly_size != "all":
            self.__set_mask_variables(anomaly_size)

    def __load_imagenet(self, imagenet_dir):
        if imagenet_dir is None:
            print("Imagenet dir not specified! Traning witout Self-Feature Enhancement with external datasets")
            self.aug_dataset = None
        else:
            try:
                self.aug_dataset = ILSVRC(ilsvrc_data_path=imagenet_dir, mean=self.mean, std=self.std, val=False)
            except Exception as ex:
                print("Imagenet was not Found! Traning witout Self-Feature Enhancement with external datasets")
                self.aug_dataset = None

    # endregion

    # region anomaly creation

    """
        source: https://github.com/YoungGod/DFC
        code was slightly adjusted

        OE_mode 0: set to black
        OE_mode 1: inpainting with opencv
        OE_mode 2: filling with sampled pixel from the original image
        OE_mode 3: filling with noise
        OE_mode 4: filling with imagenet
    """

    def __call__(self, input_img):
        if self.anomaly_size == "all":
            self.__set_mask_variables(self.anomaly_size)

        input_img = np.asarray(input_img)
        input_img = cv2.resize(input_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_normal = input_img.copy()
        img_abnormal = input_img.copy()

        # OE
        if self.aug_dataset is not None:
            oe_mode = np.random.randint(0, 5)  # OE mode
        else:
            oe_mode = np.random.randint(0, 4)  # OE mode

        # generate random anomaly mask
        original_mask = self.__generate_mask()  # [0, 1]
        rescaled_mask = resize(original_mask, (self.img_size, self.img_size), anti_aliasing=False)

        if oe_mode == 0:
            img_abnormal[rescaled_mask == 1] = 0

        # inpainting with opencv
        if oe_mode == 1:
            inpaint_mask = (rescaled_mask * 255).astype(np.uint8)
            img_abnormal = cv2.inpaint(img_abnormal, inpaint_mask, 5, cv2.INPAINT_TELEA)

        # constant fill (filling with sampled pixel from the orginal image)
        if oe_mode == 2:
            indx, indy = np.where(
                rescaled_mask > 0 if np.random.rand() > 0.4 else rescaled_mask == 0)  # prefer to sample pixel in anoamly region
            index = np.random.randint(0, len(indx))
            img_to_inpaint = np.ones_like(img_abnormal) * img_abnormal[indx[index], indy[index]]
            # alpha = 0.1 * np.random.randint(1,5)    # blending coefficient
            alpha_min = 0.1
            alpha_range = 0.4
            alpha = alpha_min + np.random.rand() * alpha_range
            img_to_inpaint = cv2.addWeighted(img_abnormal, alpha, img_to_inpaint, 1 - alpha, 0)
            img_abnormal = img_to_inpaint * rescaled_mask[:, :, np.newaxis] + img_abnormal * (
                        1 - rescaled_mask[:, :, np.newaxis])

        # noise fill
        if oe_mode == 3:
            img_to_inpaint = (np.random.rand(*img_abnormal.shape) * 255).astype(np.int16)
            img_to_inpaint = img_to_inpaint.astype(np.uint8)
            alpha_min = 0.3
            alpha_range = 0.5
            alpha = alpha_min + np.random.rand() * alpha_range
            img_to_inpaint = cv2.addWeighted(img_abnormal, alpha, img_to_inpaint, 1 - alpha, 0)
            img_abnormal = img_to_inpaint * rescaled_mask[:, :, np.newaxis] + img_abnormal * (
                        1 - rescaled_mask[:, :, np.newaxis])

        # a-blending
        if oe_mode == 4:
            # generate a image for blending (here from imagenet)
            img_to_blend = self.__generate_image_to_blend()
            alpha_min = 0.3
            alpha_range = 0.5
            alpha = alpha_min + np.random.rand() * alpha_range
            img_to_blend = cv2.addWeighted(img_abnormal, alpha, img_to_blend, 1 - alpha, 0)
            img_abnormal = img_to_blend * rescaled_mask[:, :, np.newaxis] + img_abnormal * (
                        1 - rescaled_mask[:, :, np.newaxis])

        mask_normal = np.zeros((self.mask_size, self.mask_size), dtype=np.float32)
        mask_abnormal = original_mask.astype(np.float32)

        # visualization.display_images([original_mask, rescaled_mask], ['original', 'rescaled'])
        img_normal = img_normal.astype(np.uint8)
        img_abnormal = img_abnormal.astype(np.uint8)

        return img_normal, img_abnormal, mask_normal, mask_abnormal

    def __set_mask_variables(self, size):
        if size == 'big':
            self.max_brush_width_lowervalue = self.mask_size // 32
            self.max_vertex_lowervalue = self.mask_size // 21
            self.max_brush_width_uppervalue = self.mask_size // 5
            self.max_vertex_uppervalue = self.mask_size // 16
            self.max_length = self.mask_size // 8
        elif size == 'medium':
            self.max_brush_width_lowervalue = self.mask_size // 64
            self.max_vertex_lowervalue = self.mask_size // 42
            self.max_brush_width_uppervalue = self.mask_size // 10
            self.max_vertex_uppervalue = self.mask_size // 32
            self.max_length = self.mask_size // 16
        else:
            self.max_brush_width_lowervalue = self.mask_size // 128
            self.max_vertex_lowervalue = self.mask_size // 84
            self.max_brush_width_uppervalue = self.mask_size // 20
            self.max_vertex_uppervalue = self.mask_size // 64
            self.max_length = self.mask_size // 32

    def __generate_mask(self):
        # max_brush_width_uppervalue = 48
        # max_vertex_uppervalue = 16

        max_anomaly_regions = 3  # possible max number of anomaly regions
        parts = np.random.randint(1, max_anomaly_regions + 1)

        mask = 0
        for part in range(parts):
            # brush params
            max_brush_width = np.random.randint(self.max_brush_width_lowervalue, self.max_brush_width_uppervalue)
            # max_length = 32
            max_vertex = np.random.randint(self.max_vertex_lowervalue, self.max_vertex_uppervalue)

            temp_mask = self.__np_free_form_mask(max_vertex, self.max_length, max_brush_width, max_angle=180,
                                               h=self.mask_size, w=self.mask_size)
            mask = np.logical_or(mask, temp_mask)
        return mask

    def __generate_image_to_blend(self):
        idx = int(np.random.rand() * len(self.aug_dataset))
        img_path = self.aug_dataset.img_paths[idx]
        img_to_blend = cv2.imread(img_path)
        img_to_blend = cv2.resize(img_to_blend, (self.img_size, self.img_size))
        img_to_blend = cv2.cvtColor(img_to_blend, cv2.COLOR_BGR2RGB)

        return img_to_blend

    def __np_free_form_mask(self, max_vertex, max_length, max_brush_width, max_angle, h, w):
        # start_value = self.mask_size // 13
        start_value = 10
        next_value = self.mask_size // 6

        mask = np.zeros((h, w), np.float32)
        num_vertex = np.random.randint(2, max_vertex + 1)
        start_y = np.random.randint(start_value, h - start_value)
        start_x = np.random.randint(start_value, w - start_value)
        brush_width = 0
        pre_angle = 0
        for i in range(num_vertex):
            angle = np.random.randint(max_angle + 1)
            angle = (angle / 360.0 * 2 * np.pi + pre_angle) / 2

            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(1, max_length + 1)
            brush_width = np.random.randint(self.max_brush_width_lowervalue, max_brush_width + 1) // 2 * 2
            next_y = start_y + length * np.cos(angle)
            next_x = start_x + length * np.sin(angle)

            next_y = np.maximum(np.minimum(next_y, h - next_value), next_value).astype(int)
            next_x = np.maximum(np.minimum(next_x, w - next_value), next_value).astype(int)

            cv2.line(mask, (start_y, start_x), (next_y, next_x), 1, brush_width)

            start_y, start_x = next_y, next_x
            pre_angle = angle

        return mask

    # endregion
