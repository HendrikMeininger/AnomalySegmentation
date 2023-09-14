import torch
import os
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from contextlib import contextmanager
from typing_extensions import Literal

from source.datasets.dataset import Dataset
from source.models.DFC.backbone.vgg19 import VGG19
from source.models.DFC.backbone.vgg19_s import VGG19_S

"""
Code partially based on https://github.com/YoungGod/DFC.git
"""


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

        self.feature_extraction = self.__build_feature_extractor()
        self.feature_matching = self.__build_feature_matching()
        self.optimizer = Adam(self.feature_matching.parameters(), lr=lr, weight_decay=1e-5)
        self.loss = nn.MSELoss(reduction='mean')

        self.loss_df = pd.DataFrame({'epoch': [], 'loss': [], 'loss_normal': [], 'loss_abnormal': []})
        self.val_loss_df = pd.DataFrame({'epoch': [], 'loss': [], 'loss_normal': [], 'loss_abnormal': []})
        self.eval_df = pd.DataFrame({'epoch': [], 'roc': [], 'pro': [], 'iou': []})

        self.__log_message("Training DFC model")
        self.__log_message(f"Using device: {self.device}")
        self.__log_message(f"Saving model in {self.output_dir}")

        self.__init_feat_layers()

        self.__log_message(f"Batch size: {self.batch_size}")
        self.__log_message(f"Epochs: {self.epochs}")

    def __build_feature_extractor(self):
        res = VGG19(pretrain=True,
                    gradient=False,
                    pool='avg',
                    pretrained_weights_dir=self.pretrained_weights_dir).to(self.device)
        return res

    def __build_feature_matching(self):
        res = VGG19_S(pretrain=False,
                      gradient=True,
                      pool='avg').to(self.device)
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
        with task("dataset"):
            train_loader, val_loader = self.dataset.get_train_and_val_dataloader(self.batch_size, self.train_split)
            self.__log_message(f"Using {len(train_loader.dataset)} images for training")

        with task("train"):
            self.feature_extraction.eval()
            self.feature_matching.train()

            self.__log_message("Started training...")
            loss_avg = 0.
            loss_normal_avg = 0.
            loss_abnormal_avg = 0.

            best_loss = float("inf")

            for epoch in range(self.epochs):
                self.feature_matching.train()

                for normal, abnormal, normal_mask, abnormal_mask in train_loader:
                    normal = normal.to(self.device)
                    abnormal = abnormal.to(self.device)

                    self.optimizer.zero_grad()

                    with task('normal'):
                        surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                        pred = self.feature_matching(normal, self.feat_layers)
                        loss_normal = 0
                        for feat_layer in self.feat_layers:
                            loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                        loss_normal = loss_normal / len(self.feat_layers)

                    with task('abnormal'):
                        pred = self.feature_matching(abnormal, self.feat_layers)
                        loss_abnormal = 0
                        for feat_layer in self.feat_layers:
                            loss_abnormal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                        loss_abnormal = loss_abnormal / len(self.feat_layers)

                    alpha = 1
                    loss = loss_normal + alpha * loss_abnormal
                    loss.backward()
                    self.optimizer.step()

                    # exponential moving average
                    loss_avg = loss_avg * 0.9 + float(loss.item()) * 0.1
                    loss_normal_avg = loss_normal_avg * 0.9 + float(loss_normal.item()) * 0.1
                    loss_abnormal = alpha * loss_abnormal
                    loss_abnormal_avg = loss_abnormal_avg * 0.9 + float(loss_abnormal.item()) * 0.1

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    self.__save_model("best")
                if epoch % 1 == 0:
                    self.__log_message(f"Epoch {epoch}, loss = {loss_avg:.5f}, loss_normal = {loss_normal_avg:.5f}, "
                                       f"loss_abnormal = {loss_abnormal_avg:.5f}")
                    self.loss_df.loc[len(self.loss_df)] = [epoch, loss_avg, loss_normal_avg, loss_abnormal_avg]
                    val_loss = self.__calc_validation_loss(val_loader, epoch)
                if epoch % 10 == 0 and epoch != 0:
                    self.__save_model("epoch_{}".format(epoch))
                if epoch > 100:
                    if self.early_stopping and self.__stop_early(val_loss):
                        break
                if epoch == 100:
                    self.__set_new_lr()

                torch.cuda.empty_cache()

        # save model
        self.__save_model("final")

        loss_save_path = os.path.join(self.output_dir, "loss.csv")
        self.loss_df.to_csv(loss_save_path, index=False)
        val_loss_save_path = os.path.join(self.output_dir, "val_loss.csv")
        self.val_loss_df.to_csv(val_loss_save_path, index=False)
        eval_save_path = os.path.join(self.output_dir, "eval.csv")
        self.eval_df.to_csv(eval_save_path, index=False)

        self.__log_message("Matching Net Trained.")

    # endregion

    # region private methods

    def __calc_validation_loss(self, val_data_loader, epoch):
        self.feature_matching.eval()
        loss_avg = 0.
        loss_normal_avg = 0.
        loss_abnormal_avg = 0.

        for normal, abnormal, normal_mask, abnormal_mask in val_data_loader:
            normal = normal.to(self.device)
            abnormal = abnormal.to(self.device)

            with task('normal'):
                surrogate_label_normal = self.feature_extraction(normal, self.feat_layers)
                pred = self.feature_matching(normal, self.feat_layers)
                loss_normal = 0
                for feat_layer in self.feat_layers:
                    loss_normal += self.loss(pred[feat_layer], surrogate_label_normal[feat_layer])
                loss_normal = loss_normal / len(self.feat_layers)

            with task('abnormal'):
                pred = self.feature_matching(abnormal, self.feat_layers)
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

    def __set_new_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = 2e-5

    def __save_model(self, name: str):
        save_dir = os.path.join(self.output_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, 'match.pt')
        torch.save(self.feature_matching.state_dict(), save_path)

    def __log_message(self, message):
        if self.debugging:
            print(message)

    # endregion
