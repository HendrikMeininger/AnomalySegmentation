from source.datasets import train_dataset
from source.models.patch_padim.tester import Tester
from source.models.patch_padim.trainer import Trainer
from source.utils.performance_measurement import Timer
from source.models.patch_dfc.trainer import Trainer as DFCTrainer


# !! Augmentations for self-ensembling must be adapted in the tester class.

def main():
    Timer.start_timer()

    display_predictions("wood", False)

    Timer.print_task_times()


def display_predictions(dataset_name, augmentation):
    if augmentation:
        print("Dataset: ", dataset_name, " with augmentation.")
    else:
        print("Dataset: ", dataset_name)

    tester = Tester.from_dataset_dir(dataset_dir=f"D:/mvtec_anomaly_detection/{dataset_name}",
                                     model_dir=f'D:/models/PaDiM/patch_{dataset_name}',
                                     image_size=1024,
                                     mask_size=1024,
                                     backbone="wide_resnet50",
                                     debugging=True,
                                     augmentation=augmentation)
    tester.display_predictions()


def test_dataset(dataset_name, augmentation):
    if augmentation:
        print("Dataset: ", dataset_name, " with augmentation.")
    else:
        print("Dataset: ", dataset_name)

    tester = Tester.from_dataset_dir(dataset_dir=f"D:/mvtec_anomaly_detection/{dataset_name}",
                                     model_dir=f'D:/models/PaDiM/patch_{dataset_name}',
                                     image_size=1024,
                                     mask_size=1024,
                                     backbone="wide_resnet50",
                                     debugging=True,
                                     augmentation=augmentation)
    tester.evaluate()


if __name__ == "__main__":
    main()
