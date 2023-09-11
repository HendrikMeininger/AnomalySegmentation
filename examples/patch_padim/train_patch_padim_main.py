from source.datasets import train_dataset
from source.models.PaDiM.patch_padim.trainer import Trainer
from source.utils.performance_measurement import Timer


def main():
    Timer.start_timer()

    train_model_with_dataset("bearing_small")

    Timer.log_time("Finished Training")
    Timer.print_task_times()


def train_model_with_dataset(data):
    path_to_dataset = f"E:/datasets/mvtec_anomaly_detection/{data}"
    image_paths = train_dataset.get_train_img_paths(path_to_dataset)
    trainer = Trainer(output_dir=f"E:/models/Patch-PaDiM/PP_{data}_aug_256_200_wide50",
                      image_paths=image_paths,
                      batch_size=30,
                      dataset_dir=path_to_dataset,
                      num_embeddings=200,
                      image_size=256,
                      backbone='wide_resnet50')
    trainer.train()


if __name__ == "__main__":
    main()
