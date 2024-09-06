from ultralytics import YOLO
from gradio import Interface
import gradio as gr

model_path = "./runs/detect/train/weights/best.pt"

# 加载模型
model = YOLO(model_path)  # 假设 best.pt 存放在当前目录下


def detect_objects(image):
    # 使用模型进行预测
    results = model.predict(source=image)
    if results:
        return results[0].plot()  # 返回带有标注的图像
    else:
        return "No detections found."


iface = Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),  # 输入类型为PIL图像
    outputs=gr.Image(),  # 输出也为图像
    title="Document Layoutment",
    description="Upload an image and see the layout detected by the YOLO model.",
)

# 启动 Gradio 应用
iface.launch(server_name="0.0.0.0")
