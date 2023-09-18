from source.datasets.dataset import Dataset
from source.models.PaDiM.base_padim.trainer import Trainer
from source.models.PaDiM.model import PaDiM
from source.utils.performance_measurement import Timer


def main():
    Timer.start_timer()

    path_to_dataset = ''
    output_dir = ''

    train_model_with_dataset(output_dir=output_dir, path_to_dataset=path_to_dataset)

    Timer.log_time("Finished training")
    Timer.print_task_times()


def train_model_with_dataset(output_dir: str, path_to_dataset: str):
    dataset: Dataset = Dataset(path_to_dataset=path_to_dataset, img_size=512)

    model: PaDiM = PaDiM()
    model.train(dataset=dataset, output_dir=output_dir, debugging=True, use_patches=True)


if __name__ == "__main__":
    main()
