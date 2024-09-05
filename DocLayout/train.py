from ultralytics import YOLO

model_path = "/data/bocheng/pretrained_model/yolo/yolov8n.pt"
model = YOLO(model_path)
model.train(
    task="detect",
    data="/data/bocheng/dev/mylearn/CV-Learning/DocumentLayout/yolov8_doclayney/yolov8_doclaynet.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
)
