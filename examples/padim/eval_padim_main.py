from source.datasets.dataset import Dataset
from source.models.PaDiM.base_padim.tester import Tester
from source.models.PaDiM.model import PaDiM
from source.utils.performance_measurement import Timer


def main():
    # model evaluation
    """
    Timer.start_timer()

    dataset_name = 'wood'
    eval_model(model_path=f'D:/models/Patch-PaDiM/PP_{dataset_name}_aug_256_200_wide50',
               self_ensembling=False,
               path_to_dataset=f"D:/datasets/mvtec_anomaly_detection/{dataset_name}")

    Timer.log_time("Finished evaluation")
    Timer.print_task_times()
    """

    # image prediction
    """
    img_path = 'D:/datasets/mvtec_anomaly_detection/wood/test/scratch/008.png'
    dataset_name = 'wood'
    predict_image(image_path=img_path,
                  model_path=f'D:/models/Patch-PaDiM/PP_{dataset_name}_aug_256_200_wide50/big',
                  self_ensembling=True,
                  display_prediction=True)
    """

    # display predictions
    dataset_name = 'wood'
    display_predictions(model_path=f'D:/models/Patch-PaDiM/PP_{dataset_name}_aug_256_200_wide50/big',
                        self_ensembling=False,
                        path_to_dataset=f"D:/datasets/mvtec_anomaly_detection/{dataset_name}")


def eval_model(model_path: str, self_ensembling: bool, path_to_dataset: str):
    dataset: Dataset = Dataset(path_to_dataset=path_to_dataset, img_size=1024)
    model: PaDiM = PaDiM(model_path=model_path)

    model.eval(dataset=dataset, debugging=True, self_ensembling=self_ensembling,
               rot_90=True, rot_180=True, rot_270=True, h_flip=True,
               h_flip_rot_90=True, h_flip_rot_180=True, h_flip_rot_270=True)


def predict_image(model_path: str, image_path: str, self_ensembling: bool, display_prediction: bool):
    model: PaDiM = PaDiM(model_path=model_path)

    score, binary_score = model.predict(image_path=image_path, display_prediction=display_prediction, debugging=True,
                                        self_ensembling=self_ensembling, rot_90=True, rot_180=True,
                                        rot_270=True, h_flip=True, h_flip_rot_90=True,
                                        h_flip_rot_180=True, h_flip_rot_270=True)


def display_predictions(model_path: str, path_to_dataset: str, self_ensembling: bool):
    dataset: Dataset = Dataset(path_to_dataset=path_to_dataset)
    model: PaDiM = PaDiM(model_path=model_path)

    model.display_predictions(dataset=dataset, debugging=True, self_ensembling=self_ensembling,
                              rot_90=True, rot_180=True, rot_270=True, h_flip=True,
                              h_flip_rot_90=True, h_flip_rot_180=True, h_flip_rot_270=True)


if __name__ == "__main__":
    main()
