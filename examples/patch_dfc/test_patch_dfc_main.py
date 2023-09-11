from source.models.patch_dfc.tester import Tester
from source.utils.performance_measurement import Timer


# !! Augmentations for self-ensembling must be adapted in the tester class.

def main():
    Timer.start_timer()

    test_data("leather", "textures", False)

    Timer.log_time("Without aug")

    test_data("leather", "textures", True)
    # display_predictions("tile", "textures", True)

    Timer.log_time("With aug")

    Timer.print_task_times()


def display_predictions(data, dataset_type, augmentation):
    print("Testing ", data, " augmentation: ", augmentation)
    tester = Tester.from_dataset_dir(dataset_dir=f'D:/mvtec_anomaly_detection/{data}',
                                     dataset_type=dataset_type,
                                     big_model_path=f'D:/models/DFC/patch_{data}/big/best/match.pt',
                                     medium_model_path=f'D:/models/DFC/patch_{data}/medium/best/match.pt',
                                     small_model_path=f'D:/models/DFC/patch_{data}/small/best/match.pt',
                                     debugging=True,
                                     augmentation=augmentation)
    tester.display_predictions()


def test_data(data, dataset_type, augmentation):
    print("Testing ", data, " augmentation: ", augmentation)
    tester = Tester.from_dataset_dir(dataset_dir=f'D:/datasets/mvtec_anomaly_detection/{data}',
                                     dataset_type=dataset_type,
                                     big_model_path=f'D:/models/PDFC/{data}/big/match.pt',
                                     medium_model_path=f'D:/models/PDFC/{data}/medium/match.pt',
                                     small_model_path=f'D:/models/PDFC/{data}/small/match.pt',
                                     debugging=True,
                                     augmentation=augmentation)
    tester.evaluate()


if __name__ == "__main__":
    main()
