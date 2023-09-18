from typing_extensions import Literal

from source.datasets.dataset import Dataset
from source.models.DFC.model import DFC
from source.utils.performance_measurement import Timer

_DATASET_TYPES = Literal["textures", "objects"]


def main():
    model_path = ''

    # model evaluation
    """
    Timer.start_timer()

    path_to_dataset = ''
    eval_model(model_path=model_path,
               self_ensembling=False,
               path_to_dataset=path_to_dataset,
               dataset_type='textures')

    Timer.log_time("Finished evaluation")
    Timer.print_task_times()
    """

    # image prediction
    """
    img_path = ''
    predict_image(model_path=model_path,
                  image_path=img_path,
                  dataset_type='textures',
                  self_ensembling=True,
                  display_prediction=True)
    """

    # display predictions
    path_to_dataset = ''
    display_predictions(model_path=model_path,
                        dataset_type='textures',
                        self_ensembling=False,
                        path_to_dataset=path_to_dataset)


def eval_model(model_path: str, self_ensembling: bool, path_to_dataset: str, dataset_type: _DATASET_TYPES):
    dataset: Dataset = Dataset(path_to_dataset=path_to_dataset, img_size=1024)
    model: DFC = DFC(model_path=model_path)

    model.eval(dataset=dataset, dataset_type=dataset_type, image_size=1024,
               debugging=True, self_ensembling=self_ensembling,
               rot_90=True, rot_180=True, rot_270=True, h_flip=True,
               h_flip_rot_90=True, h_flip_rot_180=True, h_flip_rot_270=True)


def predict_image(model_path: str, image_path: str, dataset_type: _DATASET_TYPES, self_ensembling: bool,
                  display_prediction: bool):
    model: DFC = DFC(model_path=model_path)

    score, binary_score = model.predict(image_path=image_path, display_prediction=display_prediction,
                                        dataset_type=dataset_type, debugging=True, image_size=1024,
                                        self_ensembling=self_ensembling, rot_90=True, rot_180=True,
                                        rot_270=True, h_flip=True, h_flip_rot_90=True,
                                        h_flip_rot_180=True, h_flip_rot_270=True)


def display_predictions(model_path: str, path_to_dataset: str, dataset_type: _DATASET_TYPES, self_ensembling: bool):
    dataset: Dataset = Dataset(path_to_dataset=path_to_dataset, img_size=1024)
    model: DFC = DFC(model_path=model_path)

    model.display_predictions(dataset=dataset, dataset_type=dataset_type, image_size=1024,
                              debugging=True, self_ensembling=self_ensembling,
                              rot_90=True, rot_180=True, rot_270=True, h_flip=True,
                              h_flip_rot_90=True, h_flip_rot_180=True, h_flip_rot_270=True)


if __name__ == "__main__":
    main()
