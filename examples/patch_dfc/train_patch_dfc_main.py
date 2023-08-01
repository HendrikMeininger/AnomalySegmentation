from source.datasets import train_dataset
from source.models.patch_dfc.trainer import Trainer


def main():
    data = 'wood'
    path_to_dataset = f"D:/datasets/mvtec_anomaly_detection/{data}"
    image_paths = train_dataset.get_train_img_paths(path_to_dataset)
    trainer = Trainer(output_dir=f"D:/models/DFC/patch_{data}",
                      image_paths=image_paths,
                      dataset_type="textures",
                      batch_size=12,
                      n_epochs=201,
                      pretrained_weights_dir=None,
                      imagenet_dir="E:/imagenet/data",
                      dataset_dir=path_to_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
