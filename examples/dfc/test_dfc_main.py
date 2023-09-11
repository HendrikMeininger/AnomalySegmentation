from source.models.DFC.tester import Tester
from source.utils.performance_measurement import Timer


# !! Augmentations for self-ensembling must be adapted in the tester class.

def main():
    Timer.start_timer()

    test_data("leather", "textures", False)

    Timer.log_time("Without aug")

    test_data("leather", "textures", True)
    # display_predictions("capsule", "objects", True)

    Timer.log_time("With aug")

    Timer.print_task_times()


def display_predictions(data, data_type, augmentation):
    print("\nData: ", data)
    tester = Tester.from_dataset_dir(dataset_dir=f"D:/mvtec_anomaly_detection/{data}",
                                     dataset_type=data_type,
                                     model_path=f'D:/models/PDFC/{data}/big/match.pt',
                                     debugging=True,
                                     augmentation=augmentation)
    tester.display_predictions()


def test_data(data, data_type, augmentation):
    print("\nData: ", data)
    tester = Tester.from_dataset_dir(dataset_dir=f"D:/datasets/mvtec_anomaly_detection/{data}",
                                     dataset_type=data_type,
                                     model_path=f'D:/models/PDFC/{data}/big/match.pt',
                                     debugging=True,
                                     augmentation=augmentation)
    tester.evaluate()


if __name__ == "__main__":
    main()
