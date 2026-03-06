from ultralytics import YOLO

def val():
    model = YOLO('runs/detect/Cow-EatOrChew-train-yolo26n-CBAM/weights/best.pt')
    # metrics = model.val()
    # print(metrics.box.map)  # map50-95
    # print(metrics.box.map50)  # map50

    fps_list = [[], [], [], []]
    for ep in range(10):
        metrics = model.val(
                            
                            imgsz=640,
                            batch=1,
                            device="0"
                            )
        fps_list[0].append(metrics.speed['preprocess'])
        fps_list[1].append(metrics.speed['inference'])
        fps_list[2].append(metrics.speed['postprocess'])
        fps_list[3].append(metrics.speed['preprocess'] + metrics.speed['inference'] + metrics.speed['postprocess'])
    print("FPS (preprocess): {:.2f}\nFPS (inference): {:.2f}\nFPS (postprocess): {:.2f}\nFPS (total): {:.2f}\nFPS (MAX): {:.2f}".format(1000 / (sum(fps_list[0]) / 10), 1000 / (sum(fps_list[1]) / 10), 1000 / (sum(fps_list[2]) / 10), 1000 / (sum(fps_list[3]) / 10) , 1000 / min(fps_list[3]) ))

if __name__ == "__main__":
    val()