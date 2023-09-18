import torch
import os
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from contextlib import contextmanager
from torchvision.transforms import Resize, InterpolationMode
from typing_extensions import Literal

from source.datasets.dataset import Dataset
from source.models.DFC.backbone.vgg19 import VGG19
from source.models.DFC.backbone.vgg19_s import VGG19_S


@contextmanager
def task(_):
    yield


class Trainer(object):

    # region init

    _DATASET_TYPES = Literal["textures", "objects"]
    _CNN_LAYERS_TEXTURES = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
    _CNN_LAYERS_OBJECTS = ("relu4_3", "relu4_4", "relu5_1", "relu5_2")

    def __init__(self, output_dir: str, dataset: Dataset, dataset_type: _DATASET_TYPES, batch_size: int = 8,
                 n_epochs: int = 201, lr: float = 2e-4, train_split: float = 0.95,
                 pretrained_weights_dir: str = None, imagenet_dir: str = None, early_stopping: bool = True,
                 patience: int = 10, debugging: bool = True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_dir = output_dir
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.learning_rate = lr
        self.train_split = train_split
        self.pretrained_weights_dir = pretrained_weights_dir
        self.imagenet_dir = imagenet_dir
        self.early_stopping = early_stopping
        self.patience = patience
        self.debugging = debugging

        self.epochs_without_improvement = 0
        self.best_val_loss = float("inf")

        self.__build_models()
        self.loss = nn.MSELoss(reduction='mean')

        self.loss_df = pd.DataFrame({'epoch': [], 'loss': [], 'loss_normal': [], 'loss_abnormal': []})
        self.val_loss_df = pd.DataFrame({'epoch': [], 'loss': [], 'loss_normal': [], 'loss_abnormal': []})
        self.eval_df = pd.DataFrame({'epoch': [], 'roc': [], 'pro': [], 'iou': []})

        self.__log_message("Training Patch DFC model")
        self.__log_message(f"Using device: {self.device}")
        self.__log_message(f"Saving model in {self.output_dir}")

        self.__init_feat_layers()

        self.__log_message(f"Batch size: {self.batch_size}")
        self.__log_message(f"Epochs: {self.epochs}")

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
        if self.dataset_type == "objects":
            self.feat_layers = self._CNN_LAYERS_OBJECTS
            self.__log_message("Using object layers.")
        elif self.dataset_type == "textures":
            self.feat_layers = self._CNN_LAYERS_TEXTURES
            self.__log_message("Using texture layers.")
        else:
            raise Exception("Unknown dataset type. Valid types: ['textures', 'objects']")

    # endregion

    # region public methods

    def train(self):
        self.feature_extraction.eval()

        if os.path.exists(os.path.join(self.output_dir, 'big', 'final')):
            self.__log_message("Big model already exists. Skipped training big model.")
        else:
            self.__train_big_patches()
        if os.path.exists(os.path.join(self.output_dir, 'medium', 'final')):
            self.__log_message("Medium model already exists. Skipped training medium model.")
        else:
            self.__train_medium_patches()
        if os.path.exists(os.path.join(self.output_dir, 'small', 'final')):
            self.__log_message("Small model already exists. Skipped training small model.")
        else:
            self.__train_small_patches()

    # endregion

    # region training

    def __train_big_patches(self):
        with task("dataset"):
            train_loader, val_loader = self.dataset.get_train_and_val_dataloader(self.batch_size, self.train_split)
            self.__log_message(f"Using {len(train_loader.dataset)} images for training")

        with task("train"):
            self.feature_extraction.eval()
            self.feature_matching_big.train()

            self.__log_message("Start training with big patches...")
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
                    self.__log_message(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, "
                                       f"loss_abnormal = {loss_abnormal_avg:.5f}")
                    self.loss_df.loc[len(self.loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]
                    val_loss = self.__calc_validation_loss(val_loader, epoch, self.feature_matching_big)
                if epoch % 10 == 0 and epoch != 0:
                    self.__save_model("big/epoch_{}".format(epoch), self.feature_matching_big)
                    self.loss_df.to_csv(loss_save_path, index=False)
                    self.val_loss_df.to_csv(val_loss_save_path, index=False)
                if epoch > 100:
                    if self.early_stopping and self.__stop_early(val_loss):
                        break
                if epoch == 100:
                    self.__set_new_lr(self.optimizer_big)

                torch.cuda.empty_cache()

        # save model
        self.__save_model("big/final", self.feature_matching_big)

        self.loss_df.to_csv(loss_save_path, index=False)
        self.val_loss_df.to_csv(val_loss_save_path, index=False)
        self.eval_df.to_csv(eval_save_path, index=False)

        self.__log_message("Matching Net for big patches Trained.")

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def __train_medium_patches(self):
        with task("dataset"):
            train_loader, val_loader = self.dataset.get_medium_patches_train_and_val_dataloader(self.batch_size,
                                                                                                self.train_split)

        with task("train"):
            self.feature_extraction.eval()
            self.feature_matching_medium.train()

            self.__log_message("Started training medium patches...")
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
                    self.__log_message(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, "
                                       f"loss_abnormal = {loss_abnormal_avg:.5f}")
                    self.loss_df.loc[len(self.loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]
                    val_loss = self.__calc_patches_validation_loss(val_loader, epoch, self.feature_matching_medium)
                if epoch % 10 == 0 and epoch != 0:
                    self.__save_model("medium/epoch_{}".format(epoch), self.feature_matching_medium)
                    self.loss_df.to_csv(loss_save_path, index=False)
                    self.val_loss_df.to_csv(val_loss_save_path, index=False)
                if epoch > 100:
                    if self.early_stopping and self.__stop_early(val_loss):
                        break
                if epoch == 100:
                    self.__set_new_lr(self.optimizer_medium)

                torch.cuda.empty_cache()

        # save model
        self.__save_model("medium/final", self.feature_matching_medium)

        self.loss_df.to_csv(loss_save_path, index=False)
        self.val_loss_df.to_csv(val_loss_save_path, index=False)

        self.__log_message("Matching Net for medium patches Trained.")

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def __train_small_patches(self):
        with task("dataset"):
            train_loader, val_loader = self.dataset.get_small_patches_train_and_val_dataloader(self.batch_size,
                                                                                               self.train_split)

        with task("train"):
            self.feature_extraction.eval()
            self.feature_matching_small.train()

            self.__log_message("Started training small patches...")
            loss_avg = 0.
            loss_normal_avg = 0.
            loss_abnormal_avg = 0.

            loss_save_path = os.path.join(self.output_dir, "small", "loss.csv")
            val_loss_save_path = os.path.join(self.output_dir, "small", "val_loss.csv")

            best_loss = float("inf")

            for epoch in range(self.epochs):
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

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    self.__save_model("small/best", self.feature_matching_small)
                if epoch % 1 == 0:
                    self.__log_message(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, "
                                       f"loss_abnormal = {loss_abnormal_avg:.5f}")
                    self.loss_df.loc[len(self.loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]
                    val_loss = self.__calc_patches_validation_loss(val_loader, epoch, self.feature_matching_small)
                if epoch % 10 == 0 and epoch != 0:
                    self.__save_model("small/epoch_{}".format(epoch), self.feature_matching_small)
                    self.loss_df.to_csv(loss_save_path, index=False)
                    self.val_loss_df.to_csv(val_loss_save_path, index=False)
                if epoch > 100:
                    if self.early_stopping and self.__stop_early(val_loss):
                        break
                if epoch == 100:
                    self.__set_new_lr(self.optimizer_small)

                torch.cuda.empty_cache()

        # save model
        self.__save_model("small/final", self.feature_matching_small)

        self.loss_df.to_csv(loss_save_path, index=False)
        self.val_loss_df.to_csv(val_loss_save_path, index=False)

        self.__log_message("Matching Net for small patches Trained.")

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

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

    def __log_message(self, message):
        if self.debugging:
            print(message)

    # endregion
