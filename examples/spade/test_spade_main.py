from source.models.SPADE.trainer import Trainer
from source.models.SPADE.tester import Tester
from source.utils.performance_measurement import Timer


def main():
    Timer.start_timer()

    # test_model("leather", False, None, None, None, [])
    # test_model("leather", True, True, True, True, [90, 180, 270])
    display_predictions("leather", False, None, None, None, [])

    Timer.log_time("Finished testing")
    Timer.print_task_times()


def display_predictions(dataset_name, augmentation, v_flip, h_flip, rotate, rotation_degrees):
    dataset_dir = f"D:/mvtec_anomaly_detection/{dataset_name}"
    if augmentation:
        print("Dataset: ", dataset_name, " with TPE.")
        dataset_name += "_SE"
    else:
        print("Dataset: ", dataset_name)

    Timer.start_timer()
    tester = Tester(dataset_dir=dataset_dir,
                    model_path=f'D:/models/SPADE/{dataset_name}',
                    image_size=256,
                    mask_size=1024,
                    debugging=True,
                    augmentation=augmentation,
                    top_k=20,
                    v_flip=v_flip,
                    h_flip=h_flip,
                    rotate=rotate,
                    rotation_degrees=rotation_degrees)
    tester.display_predictions()


def test_model(dataset_name, augmentation, v_flip, h_flip, rotate, rotation_degrees):
    dataset_dir = f"D:/mvtec_anomaly_detection/{dataset_name}"
    if augmentation:
        print("Dataset: ", dataset_name, " with TPE.")
        dataset_name += "_SE"
    else:
        print("Dataset: ", dataset_name)

    Timer.start_timer()
    tester = Tester(dataset_dir=dataset_dir,
                    model_path=f'D:/models/SPADE/{dataset_name}',
                    image_size=256,
                    mask_size=1024,
                    debugging=True,
                    augmentation=augmentation,
                    top_k=20,
                    v_flip=v_flip,
                    h_flip=h_flip,
                    rotate=rotate,
                    rotation_degrees=rotation_degrees)

    tester.evaluate()
    Timer.log_time("Finished Evaluation")
    Timer.print_task_times()


if __name__ == "__main__":
    main()
