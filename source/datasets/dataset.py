from torch.utils.data import DataLoader

from source.datasets.train_dataset import TrainDataset
from source.datasets.patch_train_dataset import PatchTrainDataset


class Dataset(object):

    def __init__(self):
        pass

    def get_train_dataloader(self) -> DataLoader:
        train_data = TrainDataset(image_paths=self.image_paths, img_size=self.image_size,
                                  imagenet_dir="E:/imagenet/data", horizontal_flip=False, vertical_flip=False,
                                  rotate=False, adjust_sharpness=False, auto_contrast=False, color_jitter=False)
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def get_patch_train_dataloader(self) -> DataLoader:
        patch_train_data = PatchTrainDataset(image_paths=self.image_paths, patch_size=256, image_size=512,
                                             imagenet_dir="",
                                             horizontal_flip=True, vertical_flip=True, rotate=True,
                                             adjust_sharpness=False, auto_contrast=False, color_jitter=False,
                                             create_anomaly=False)
        train_loader = DataLoader(dataset=patch_train_data, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def get_test_dataloader(self) -> DataLoader:
        pass
