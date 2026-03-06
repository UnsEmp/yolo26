from ultralytics import YOLO


def train():
    # Load a pretrained YOLO12n model
    model = YOLO("datasets/Cow-eat-chew-behaviour/yolo26n_8.yaml")
    # model = YOLO("runs/detect/Cow-EatOrChew-train-yolo26n-C3k2Star/weights/best.pt")

    # Train the model  for  epochs
    model.train(
        data="datasets/Cow-eat-chew-behaviour/CHVI.yaml",  # Path to dataset configuration file
        epochs=300,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        mpdiou=False,  # Use MPDIoU loss for bounding box regression
        # optimizer="AdamW",  # Optimizer to use (e.g., 'SGD', 'Adam', 'AdamW')
        resume=True,
        workers=4,
        # patience=50,
        batch=16,
        name="Cow-EatOrChew-train-yolo26n-MSAC2PSA",  # Name for the training run, used for saving results
        save_period=20,  # 保存模型检查点的频率，以 epoch 为单位指定。值为 -1 时禁用此功能。适用于在长时间训练期间保存临时模型。
        plots=True,  # 生成并保存训练和验证指标的图表，以及预测示例，从而 提供对模型性能和学习进度的可视化见解
    )

    # calculate average FPS after training
    fps_list = [[], [], [], []]
    for ep in range(10):
        metrics = model.val(
            project="runs/detect/Cow-EatOrChew-train-yolo26n-MSAC2PSA",
            data="datasets/Cow-eat-chew-behaviour/CHVI.yaml",
            imgsz=640,
            batch=16,
            device="0",
        )
        fps_list[0].append(metrics.speed["preprocess"])
        fps_list[1].append(metrics.speed["inference"])
        fps_list[2].append(metrics.speed["postprocess"])
        fps_list[3].append(metrics.speed["preprocess"] + metrics.speed["inference"] + metrics.speed["postprocess"])
    print(
        f"FPS (preprocess): {1000 / (sum(fps_list[0]) / 10):.2f}\nFPS (inference): {1000 / (sum(fps_list[1]) / 10):.2f}\nFPS (postprocess): {1000 / (sum(fps_list[2]) / 10):.2f}\nFPS (total): {1000 / (sum(fps_list[3]) / 10):.2f}\nFPS (MAX): {1000 / min(fps_list[3]):.2f}"
    )

    # Evaluate the model's performance on the validation set
    # metrics = model.val(project='runs/detect/Cow-EatOrChew-train-yolo26n',
    #                     data="datasets/Cow-eat-chew-behaviour/CHVI.yaml",
    #                     imgsz=640,
    #                     batch=1,
    #                     device="0"
    #                     )

    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list contains map50-95 of each category

    # Perform object detection on an image
    # results = model("path/to/image.jpg")  # Predict on an image
    # results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    # path = model.export(format="onnx")  # Returns the path to the exported model


if __name__ == "__main__":
    # start train ~
    train()
