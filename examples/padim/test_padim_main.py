from source.models.PaDiM.tester import Tester
from source.utils.performance_measurement import Timer


# !! Augmentations for self-ensembling must be adapted in the tester class.

def main():
    Timer.start_timer()

    # test_dataset('pill', False)
    display_predictions('pill', False)

    Timer.log_time("Finished testing")

    Timer.print_task_times()


def display_predictions(dataset_name, augmentation):
    tester = Tester.from_dataset_dir(dataset_dir=f"D:/mvtec_anomaly_detection/{dataset_name}",
                                     model_path=f'D:/models/PaDiM/{dataset_name}',
                                     image_size=256,
                                     mask_size=1024,
                                     backbone="wide_resnet50",
                                     debugging=True,
                                     augmentation=augmentation)
    tester.display_predictions()


def test_dataset(dataset_name, augmentation):
    if augmentation:
        tester = Tester.from_dataset_dir(dataset_dir=f"D:/mvtec_anomaly_detection/{dataset_name}",
                                         model_path=f'D:/models/PaDiM/{dataset_name}_SE',
                                         image_size=256,
                                         mask_size=1024,
                                         backbone="wide_resnet50",
                                         debugging=True,
                                         augmentation=augmentation)
    else:
        tester = Tester.from_dataset_dir(dataset_dir=f"D:/mvtec_anomaly_detection/{dataset_name}",
                                         model_path=f'D:/models/PaDiM/{dataset_name}',
                                         image_size=256,
                                         mask_size=1024,
                                         backbone="wide_resnet50",
                                         debugging=True,
                                         augmentation=augmentation)

    tester.evaluate()


if __name__ == "__main__":
    main()
