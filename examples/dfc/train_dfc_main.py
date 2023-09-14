from source.datasets.dataset import Dataset
from source.models.DFC.model import DFC
from source.utils.performance_measurement import Timer


def main():
    Timer.start_timer()

    data = 'carpet'
    path_to_dataset = f"D:/datasets/mvtec_anomaly_detection/{data}"
    output_dir = 'C:/Vision4Quality/testmodel'

    train_model_with_dataset(output_dir=output_dir, path_to_dataset=path_to_dataset)

    Timer.log_time("Finished training")
    Timer.print_task_times()


def train_model_with_dataset(output_dir: str, path_to_dataset: str):
    dataset: Dataset = Dataset(path_to_dataset=path_to_dataset, img_size=256, self_supervised_training=True)

    model: DFC = DFC()
    model.train(dataset=dataset, dataset_type='textures', output_dir=output_dir,
                debugging=True, use_patches=True, epochs=3)


if __name__ == "__main__":
    main()
