from ultralytics import YOLO

model_path = "/data/bocheng/pretrained_model/yolo/yolov8n.pt"
model = YOLO(model_path)
model.train(
    task="detect",
    data="./yolov8_math_formula.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    # resume=True,
    device=[0],
)
