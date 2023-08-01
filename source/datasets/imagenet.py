import torch
import torch.utils.data
import numpy as np
import os
import glob
from PIL import Image
import random
import torchvision.transforms as transforms


# source: https://github.com/YoungGod/DFC
# code was slightly adjusted

class ILSVRC(torch.utils.data.Dataset):
    def __init__(self, ilsvrc_data_path, mean, std, val=True):

        if val:
            synset_dir = os.path.join(ilsvrc_data_path, 'val')
        else:
            synset_dir = os.path.join(ilsvrc_data_path, 'train')
        self.img_paths = self.get_image_paths(synset_dir)

        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    #         self.transform = transforms.Compose([transforms.RandomCrop(), transforms.ToTensor()])
    # self.transform = transform    # we define transforms outside the data set

    def __getitem__(self, index):
        path = self.img_paths[index]
        # print(sample)
        # read and transform
        img = self.preprocess_image(path)  # PIL image
        # print(img)
        # img=np.transpose(img,(2,0,1))
        # img = torch.from_numpy(img)
        img = np.array(img)
        img = self.transform(img)

        return img
        # return img    # only use the img

    def __len__(self):
        return len(self.img_paths)

    def preprocess_image(self, image_path):
        """ It reads an image, it resize it to have the lowest dimesnion of 256px,
            it randomly choose a 224x224 crop inside the resized image and normilize the numpy 
            array subtracting the ImageNet training set mean
            Args:
                images_path: path of the image
            Returns:
                cropped_im_array: the numpy array of the image normalized [width, height, channels]
        """
        # IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format
        # print(image_path)
        img = Image.open(image_path).convert('RGB')
        # print(img)
        # resize of the image (setting lowest dimension to 256px)
        if img.size[0] < img.size[1]:
            h = int(float(256 * img.size[1]) / img.size[0])
            img = img.resize((256, h), Image.ANTIALIAS)  # 抗锯齿
        else:
            w = int(float(256 * img.size[0]) / img.size[1])
            img = img.resize((w, 256), Image.ANTIALIAS)

        # in case when the image size < (256, 256)
        if img.size[0] < 256 or img.size[1] < 256:
            img = img.resize((256, 256), Image.ANTIALIAS)

        # random 244x224 patch
        x = random.randint(0, img.size[0] - 224)
        y = random.randint(0, img.size[1] - 224)
        img_cropped = img.crop((x, y, x + 224, y + 224))

        # data augmentation: flip left right
        if random.randint(0, 1) == 1:
            img_cropped = img_cropped.transpose(Image.FLIP_LEFT_RIGHT)

        # cropped_im_array = np.array(img_cropped, dtype=np.float32)
        #
        # for i in range(3):
        #     cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]
        #
        # return cropped_im_array/225
        return img_cropped

    def get_image_paths(self, data_path):
        img_paths = []
        pattern = data_path + "/*/*.JPEG"
        for path in glob.glob(pattern):
            img_paths.append(path)

        return img_paths
