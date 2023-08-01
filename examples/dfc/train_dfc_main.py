from source.datasets import train_dataset
from source.models.DFC.trainer import Trainer


def main():
    path_to_dataset = "E:/datasets/mvtec_anomaly_detection/capsule"
    image_paths = train_dataset.get_train_img_paths(path_to_dataset)
    trainer = Trainer(output_dir="E:/models/DFC/capsule",
                      image_paths=image_paths,
                      dataset_type="objects",
                      batch_size=12,
                      n_epochs=301,
                      pretrained_weights_dir=None,
                      imagenet_dir="E:/imagenet/data",
                      dataset_dir=path_to_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
