from source.models.SPADE.base_spade.trainer import Trainer
from source.utils.performance_measurement import Timer


def main():
    Timer.start_timer()
    train_model("carpet")
    train_model("grid")
    train_model("leather")
    train_model("tile")
    train_model("wood")

    train_model("bottle")
    train_model("cable")
    train_model("capsule")
    train_model("hazelnut")
    train_model("metal_nut")
    train_model("pill")
    train_model("screw")
    train_model("toothbrush")
    train_model("transistor")
    train_model("zipper")
    Timer.log_time("Finished Training")
    Timer.print_task_times()


def train_model(data):
    path_to_dataset = f"E:/datasets/mvtec_anomaly_detection/{data}"
    trainer = Trainer(output_dir=f"E:/models/SPADE/SPADE_{data}",
                      path_to_dataset=path_to_dataset,
                      image_size=256,
                      batch_size=32)
    trainer.train()

    del trainer
    print(f"Finished training {data}")


if __name__ == "__main__":
    main()
