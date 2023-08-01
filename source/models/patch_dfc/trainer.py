import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from contextlib import contextmanager

from torchvision.transforms import Resize, InterpolationMode

from source.datasets.patch_train_dataset import PatchTrainDataset
from source.datasets.train_dataset import TrainDataset
from source.models.DFC.tester import Tester
from source.models.DFC_backbone.vgg19 import VGG19
from source.models.DFC_backbone.vgg19_s import VGG19_S
from source.utils.performance_measurement import Timer


@contextmanager
def task(_):
    yield


class Trainer(object):

    # region init

    def __init__(self, output_dir, image_paths, dataset_type,
                 dataset_dir,
                 batch_size=8, n_epochs=201, lr=2e-4,
                 pretrained_weights_dir=None, imagenet_dir=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_dir = output_dir
        self.image_paths = image_paths
        self.dataset_type = dataset_type
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.learning_rate = lr
        self.pretrained_weights_dir = pretrained_weights_dir
        self.imagenet_dir = imagenet_dir

        self.patience = 10
        self.epochs_without_improvement = 0
        self.best_val_loss = float("inf")

        self.__split_image_paths(image_paths)

        self.__build_models()
        self.loss = nn.MSELoss(reduction='mean')

        self.loss_df = pd.DataFrame({'epoch': [], 'loss': [], 'loss_normal': [], 'loss_abnormal': []})
        self.val_loss_df = pd.DataFrame({'epoch': [], 'loss': [], 'loss_normal': [], 'loss_abnormal': []})
        self.eval_df = pd.DataFrame({'epoch': [], 'roc': [], 'pro': [], 'iou': []})

        print("Training Patch DFC model")
        print("Using device: ", self.device)
        print("Saving model in ", self.output_dir)

        self.__init_feat_layers()

        print("Using ", len(self.image_paths), " images for training")
        print("Batch size: ", self.batch_size)
        print("Epochs: ", self.epochs)

    def __build_models(self):
        self.feature_extraction = self.__build_feature_extractor()

        self.feature_matching_big = self.__build_feature_matching()
        self.feature_matching_medium = self.__build_feature_matching()
        self.feature_matching_small = self.__build_feature_matching()

        self.optimizer_big = Adam(self.feature_matching_big.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.optimizer_medium = Adam(self.feature_matching_medium.parameters(), lr=self.learning_rate,
                                     weight_decay=1e-5)
        self.optimizer_small = Adam(self.feature_matching_small.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def __build_feature_extractor(self):
        res = VGG19(pretrain=True, gradient=False, pool='avg',
                    pretrained_weights_dir=self.pretrained_weights_dir).to(self.device)
        return res

    def __build_feature_matching(self):
        res = VGG19_S(pretrain=False, gradient=True, pool='avg').to(self.device)
        return res

    def __init_feat_layers(self):
        cnn_layers_textures = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
        cnn_layers_objects = ("relu4_3", "relu4_4", "relu5_1", "relu5_2")

        if self.dataset_type == "objects":
            self.feat_layers = cnn_layers_objects
            print("Using object layers.")
        elif self.dataset_type == "textures":
            self.feat_layers = cnn_layers_textures
            print("Using texture layers.")

    def __split_image_paths(self, image_paths):
        n = len(image_paths)
        n_train = int(n * 0.95)

        self.train_image_paths = image_paths[:n_train]
        self.val_image_paths = image_paths[n_train:]

    # endregion

    # region public methods

    def train(self):
        self.feature_extraction.eval()

        if os.path.exists(os.path.join(self.output_dir, 'big', 'final')):
            print("Big model already exists. Skipped training big model.")
        else:
            self.__train_big_patches()
        if os.path.exists(os.path.join(self.output_dir, 'medium', 'final')):
            print("Medium model already exists. Skipped training medium model.")
        else:
            self.__train_medium_patches()
        if os.path.exists(os.path.join(self.output_dir, 'small', 'final')):
            print("Small model already exists. Skipped training small model.")
        else:
            self.__train_small_patches()

    # endregion

    # region training

    def __train_big_patches(self):
        Timer.start_timer()

        with task("dataset"):
            train_data = TrainDataset(image_paths=self.train_image_paths, img_size=256,
                                      mask_size=1024, imagenet_dir=self.imagenet_dir,
                                      horizontal_flip=True, vertical_flip=True, rotate=True,
                                      adjust_sharpness=False, auto_contrast=False, color_jitter=False,
                                      anomaly_size='all', method='dfc')
            train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)

            val_data = TrainDataset(image_paths=self.val_image_paths, img_size=256,
                                    mask_size=1024, imagenet_dir=self.imagenet_dir,
                                    horizontal_flip=False, vertical_flip=False, rotate=False,
                                    adjust_sharpness=False, auto_contrast=False, color_jitter=False,
                                    anomaly_size='all', method='all')
            val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

        with task("train"):
            self.feature_extraction.eval()
            self.feature_matching_big.train()

            print("Start training with big patches...")
            loss_avg = 0.
            loss_normal_avg = 0.
            loss_abnormal_avg = 0.

            loss_save_path = os.path.join(self.output_dir, "big", "loss.csv")
            val_loss_save_path = os.path.join(self.output_dir, "big", "val_loss.csv")
            eval_save_path = os.path.join(self.output_dir, "eval.csv")

            best_loss = float("inf")

            for epoch in range(self.epochs):
                self.feature_matching_big.train()

                for normal, abnormal, normal_mask, abnormal_mask in train_loader:
                    normal = normal.to(self.device)
                    abnormal = abnormal.to(self.device)

                    self.optimizer_big.zero_grad()

                    with task('normal'):
                        surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                        pred = self.feature_matching_big(normal, self.feat_layers)
                        loss_normal = 0
                        for feat_layer in self.feat_layers:
                            loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                        loss_normal = loss_normal / len(self.feat_layers)

                    with task('abnormal'):
                        pred = self.feature_matching_big(abnormal, self.feat_layers)
                        loss_abnormal = 0
                        for feat_layer in self.feat_layers:
                            loss_abnormal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                        loss_abnormal = loss_abnormal / len(self.feat_layers)

                    alpha = 1
                    loss = loss_normal + alpha * loss_abnormal
                    loss.backward()
                    self.optimizer_big.step()

                    # exponential moving average
                    loss_avg = loss_avg * 0.9 + float(loss.item()) * 0.1
                    loss_normal_avg = loss_normal_avg * 0.9 + float(loss_normal.item()) * 0.1
                    loss_abnormal = alpha * loss_abnormal
                    loss_abnormal_avg = loss_abnormal_avg * 0.9 + float(loss_abnormal.item()) * 0.1

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    self.__save_model("big/best", self.feature_matching_big)
                if epoch % 1 == 0:
                    print(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, "
                          f"loss_abnormal = {loss_abnormal_avg:.5f}")
                    self.loss_df.loc[len(self.loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]
                    val_loss = self.__calc_validation_loss(val_loader, epoch, self.feature_matching_big)
                if epoch % 10 == 0:
                    self.__save_model("big/epoch_{}".format(epoch), self.feature_matching_big)
                    self.loss_df.to_csv(loss_save_path, index=False)
                    self.val_loss_df.to_csv(val_loss_save_path, index=False)

                    self.eval_df.to_csv(eval_save_path, index=False)
                if epoch > 100:
                    if self.__stop_early(val_loss):
                        break
                if epoch == 100:
                    self.__set_new_lr(self.optimizer_big)

                torch.cuda.empty_cache()

        # save model
        self.__save_model("big/final", self.feature_matching_big)

        self.loss_df.to_csv(loss_save_path, index=False)
        self.val_loss_df.to_csv(val_loss_save_path, index=False)
        self.eval_df.to_csv(eval_save_path, index=False)

        Timer.log_time("Trained Matching Net for big patches")
        print("Matching Net for big patches Trained.")

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        Timer.print_task_times()

    def __train_medium_patches(self):
        Timer.start_timer()

        with task("dataset"):
            patch_train_data = PatchTrainDataset(image_paths=self.train_image_paths, patch_size=256, image_size=512,
                                                 imagenet_dir=self.imagenet_dir,
                                                 horizontal_flip=True, vertical_flip=True, rotate=True,
                                                 adjust_sharpness=False, auto_contrast=False, color_jitter=False)
            train_loader = DataLoader(dataset=patch_train_data, batch_size=self.batch_size, shuffle=True)

            patch_val_data = PatchTrainDataset(image_paths=self.train_image_paths, patch_size=256, image_size=512,
                                               imagenet_dir=self.imagenet_dir,
                                               horizontal_flip=False, vertical_flip=False, rotate=False,
                                               adjust_sharpness=False, auto_contrast=False, color_jitter=False)
            val_loader = DataLoader(dataset=patch_val_data, batch_size=1, shuffle=False)

        with task("train"):
            self.feature_extraction.eval()
            self.feature_matching_medium.train()

            print("Started training medium patches...")
            loss_avg = 0.
            loss_normal_avg = 0.
            loss_abnormal_avg = 0.

            loss_save_path = os.path.join(self.output_dir, "medium", "loss.csv")
            val_loss_save_path = os.path.join(self.output_dir, "medium", "val_loss.csv")

            best_loss = float("inf")

            for epoch in range(self.epochs):
                self.feature_matching_medium.train()

                for normals, abnormals, normal_masks, abnormal_masks in train_loader:  # for all train images
                    for normal, abnormal, normal_mask, abnormal_mask in zip(normals, abnormals, normal_masks,
                                                                            abnormal_masks):
                        normal = normal.to(self.device)
                        abnormal = abnormal.to(self.device)

                        self.optimizer_medium.zero_grad()

                        with task('normal'):
                            surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                            pred = self.feature_matching_medium(normal, self.feat_layers)
                            loss_normal = 0
                            for feat_layer in self.feat_layers:
                                loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                            loss_normal = loss_normal / len(self.feat_layers)

                        with task('abnormal'):
                            pred = self.feature_matching_medium(abnormal, self.feat_layers)
                            loss_abnormal = 0
                            for feat_layer in self.feat_layers:
                                loss_abnormal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                            loss_abnormal = loss_abnormal / len(self.feat_layers)

                        alpha = 1
                        loss = loss_normal + alpha * loss_abnormal
                        loss.backward()
                        self.optimizer_medium.step()

                        # exponential moving average
                        loss_avg = loss_avg * 0.9 + float(loss.item()) * 0.1
                        loss_normal_avg = loss_normal_avg * 0.9 + float(loss_normal.item()) * 0.1
                        loss_abnormal = alpha * loss_abnormal
                        loss_abnormal_avg = loss_abnormal_avg * 0.9 + float(loss_abnormal.item()) * 0.1

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    self.__save_model("medium/best", self.feature_matching_medium)
                if epoch % 1 == 0:
                    print(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, "
                          f"loss_abnormal = {loss_abnormal_avg:.5f}")
                    self.loss_df.loc[len(self.loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]
                    val_loss = self.__calc_patches_validation_loss(val_loader, epoch, self.feature_matching_medium)
                if epoch % 10 == 0:
                    self.__save_model("medium/epoch_{}".format(epoch), self.feature_matching_medium)
                    self.loss_df.to_csv(loss_save_path, index=False)
                    self.val_loss_df.to_csv(val_loss_save_path, index=False)
                if epoch > 100:
                    if self.__stop_early(val_loss):
                        break
                if epoch == 100:
                    self.__set_new_lr(self.optimizer_medium)

                torch.cuda.empty_cache()

        # save model
        self.__save_model("medium/final", self.feature_matching_medium)

        self.loss_df.to_csv(loss_save_path, index=False)
        self.val_loss_df.to_csv(val_loss_save_path, index=False)

        Timer.log_time("Trained Matching Net for medium patches")
        print("Matching Net for medium patches Trained.")

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        Timer.print_task_times()

    def __train_small_patches(self):
        Timer.start_timer()

        with task("dataset"):
            patch_train_data = PatchTrainDataset(image_paths=self.train_image_paths, patch_size=256, image_size=1024,
                                                 imagenet_dir=self.imagenet_dir,
                                                 horizontal_flip=True, vertical_flip=True, rotate=True,
                                                 adjust_sharpness=False, auto_contrast=False, color_jitter=False)
            train_loader = DataLoader(dataset=patch_train_data, batch_size=self.batch_size, shuffle=True)

            patch_val_data = PatchTrainDataset(image_paths=self.train_image_paths, patch_size=256, image_size=1024,
                                               imagenet_dir=self.imagenet_dir,
                                               horizontal_flip=False, vertical_flip=False, rotate=False,
                                               adjust_sharpness=False, auto_contrast=False)
            val_loader = DataLoader(dataset=patch_val_data, batch_size=1, shuffle=False)

        with task("train"):
            Timer.start_timer()

            self.feature_extraction.eval()
            self.feature_matching_small.train()

            print("Started training small patches...")
            loss_avg = 0.
            loss_normal_avg = 0.
            loss_abnormal_avg = 0.

            loss_save_path = os.path.join(self.output_dir, "small", "loss.csv")
            val_loss_save_path = os.path.join(self.output_dir, "small", "val_loss.csv")

            best_loss = float("inf")

            for epoch in range(41, self.epochs):
                self.feature_matching_small.train()

                for normals, abnormals, normal_masks, abnormal_masks in train_loader:  # for all train images
                    for normal, abnormal, normal_mask, abnormal_mask in zip(normals, abnormals, normal_masks,
                                                                            abnormal_masks):
                        normal = normal.to(self.device)
                        abnormal = abnormal.to(self.device)

                        self.optimizer_small.zero_grad()

                        with task('normal'):
                            surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                            pred = self.feature_matching_small(normal, self.feat_layers)
                            loss_normal = 0
                            for feat_layer in self.feat_layers:
                                loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                            loss_normal = loss_normal / len(self.feat_layers)

                        with task('abnormal'):
                            pred = self.feature_matching_small(abnormal, self.feat_layers)
                            loss_abnormal = 0
                            for feat_layer in self.feat_layers:
                                loss_abnormal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                            loss_abnormal = loss_abnormal / len(self.feat_layers)

                        alpha = 1
                        loss = loss_normal + alpha * loss_abnormal
                        loss.backward()
                        self.optimizer_small.step()

                        # exponential moving average
                        loss_avg = loss_avg * 0.9 + float(loss.item()) * 0.1
                        loss_normal_avg = loss_normal_avg * 0.9 + float(loss_normal.item()) * 0.1
                        loss_abnormal = alpha * loss_abnormal
                        loss_abnormal_avg = loss_abnormal_avg * 0.9 + float(loss_abnormal.item()) * 0.1

                Timer.log_time(f"Finished epoch {epoch}")
                Timer.print_task_times()

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    self.__save_model("small/best", self.feature_matching_small)
                if epoch % 1 == 0:
                    print(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, "
                          f"loss_abnormal = {loss_abnormal_avg:.5f}")
                    self.loss_df.loc[len(self.loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]
                    val_loss = self.__calc_patches_validation_loss(val_loader, epoch, self.feature_matching_small)
                if epoch % 10 == 0:
                    self.__save_model("small/epoch_{}".format(epoch), self.feature_matching_small)
                    self.loss_df.to_csv(loss_save_path, index=False)
                    self.val_loss_df.to_csv(val_loss_save_path, index=False)
                if epoch > 100:
                    if self.__stop_early(val_loss):
                        break
                if epoch == 100:
                    self.__set_new_lr(self.optimizer_small)

                torch.cuda.empty_cache()

        # save model
        self.__save_model("small/final", self.feature_matching_small)

        self.loss_df.to_csv(loss_save_path, index=False)
        self.val_loss_df.to_csv(val_loss_save_path, index=False)

        Timer.log_time("Trained Matching Net for small patches")
        print("Matching Net for small patches Trained.")

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        Timer.print_task_times()

    # endregion

    # region private methods

    def __calc_validation_loss(self, val_data_loader, epoch, model):
        model.eval()
        loss_avg = 0.
        loss_normal_avg = 0.
        loss_abnormal_avg = 0.

        for normal, abnormal, normal_mask, abnormal_mask in val_data_loader:
            normal = Resize(size=256, interpolation=InterpolationMode.BILINEAR)(normal).to(self.device)
            abnormal = Resize(size=256, interpolation=InterpolationMode.BILINEAR)(abnormal).to(self.device)

            with task('normal'):
                surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                pred = model(normal, self.feat_layers)
                loss_normal = 0
                for feat_layer in self.feat_layers:
                    loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                loss_normal = loss_normal / len(self.feat_layers)

            with task('abnormal'):
                pred = model(abnormal, self.feat_layers)
                loss_abnormal = 0
                for feat_layer in self.feat_layers:
                    loss_abnormal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                loss_abnormal = loss_abnormal / len(self.feat_layers)

            alpha = 1
            loss = loss_normal + alpha * loss_abnormal

            # exponential moving average
            loss_avg = loss_avg * 0.9 + float(loss.item()) * 0.1
            loss_normal_avg = loss_normal_avg * 0.9 + float(loss_normal.item()) * 0.1
            loss_abnormal = alpha * loss_abnormal
            loss_abnormal_avg = loss_abnormal_avg * 0.9 + float(loss_abnormal.item()) * 0.1

        self.val_loss_df.loc[len(self.val_loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]

        return loss_avg

    def __calc_patches_validation_loss(self, val_data_loader, epoch, model):
        model.eval()
        loss_avg = 0.
        loss_normal_avg = 0.
        loss_abnormal_avg = 0.

        for normals, abnormals, normal_masks, abnormal_masks in val_data_loader:  # for all train images
            for normal, abnormal, normal_mask, abnormal_mask in zip(normals, abnormals, normal_masks,
                                                                    abnormal_masks):
                normal = Resize(size=256, interpolation=InterpolationMode.BILINEAR)(normal).to(self.device)
                abnormal = Resize(size=256, interpolation=InterpolationMode.BILINEAR)(abnormal).to(self.device)

                with task('normal'):
                    surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                    pred = model(normal, self.feat_layers)
                    loss_normal = 0
                    for feat_layer in self.feat_layers:
                        loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                    loss_normal = loss_normal / len(self.feat_layers)

                with task('abnormal'):
                    pred = model(abnormal, self.feat_layers)
                    loss_abnormal = 0
                    for feat_layer in self.feat_layers:
                        loss_abnormal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                    loss_abnormal = loss_abnormal / len(self.feat_layers)

                alpha = 1
                loss = loss_normal + alpha * loss_abnormal

                # exponential moving average
                loss_avg = loss_avg * 0.9 + float(loss.item()) * 0.1
                loss_normal_avg = loss_normal_avg * 0.9 + float(loss_normal.item()) * 0.1
                loss_abnormal = alpha * loss_abnormal
                loss_abnormal_avg = loss_abnormal_avg * 0.9 + float(loss_abnormal.item()) * 0.1

        self.val_loss_df.loc[len(self.val_loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]

        return loss_avg

    def __stop_early(self, loss) -> bool:
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.epochs_without_improvement = 0

            return False

        self.epochs_without_improvement += 1

        return self.epochs_without_improvement > self.patience

    def __set_new_lr(self, optimizer):
        for g in optimizer.param_groups:
            g['lr'] = 2e-5

    def __save_model(self, name: str, model):
        save_dir = os.path.join(self.output_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, 'match.pt')
        torch.save(model.state_dict(), save_path)

    # endregion
