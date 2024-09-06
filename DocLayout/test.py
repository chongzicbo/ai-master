from ultralytics import YOLO
from pprint import pprint
from pathlib import Path

model_path = "./runs/detect/train/weights/best.pt"
model = YOLO(model_path)
image_path = Path("./data/test.png")
results = model.predict(source=image_path.absolute())
output_dir = Path("./runs/detect/test/output")
output_dir.mkdir(parents=True, exist_ok=True)
if results:
    result = results[0]
    save_path = output_dir / image_path
    result.save(output_dir / image_path.name)
    pprint(f"Detection result saved to {save_path}")
else:
    pprint("No detection results found.")
