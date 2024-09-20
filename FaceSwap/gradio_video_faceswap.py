# -*-coding:utf-8 -*-

"""
# File       : faceswap_video_gradio.py
# Time       ：2023/6/13 12:24
# Author     ：chengbo
# version    ：python 3.8
# Description：使用Gradio实现视频上传、换脸、显示换脸后的视频并提供下载按钮
"""
import sys
import os
import shutil

sys.path.append("..")
import faceswap_insightface
from video_process import video_images
import insightface
from insightface.app import FaceAnalysis
import gradio as gr

# 指定中间目录和换脸后的图片目录的根目录
BASE_DIR = "/data/bocheng/mywork/faceswap/"
os.makedirs(BASE_DIR, exist_ok=True)


def swapface(
    input_video_path: str,
    target_face_path: str,
    swapper: insightface.model_zoo.inswapper.INSwapper,
    app: FaceAnalysis,
):
    """
    单人的视频换脸
    1.视频抽取帧
    2.每张图片进行人脸转换
    3.对人脸转换后的图片进行视频合成
    :param input_video_path:输入视频路径
    :param target_face_path:包含目标face的图片
    :return: 换脸后的视频路径
    """
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_frames_dir = os.path.join(BASE_DIR, video_name, "output_frames")
    swap_image_dir = os.path.join(BASE_DIR, video_name, "swapped_frames")
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    if not os.path.exists(swap_image_dir):
        os.makedirs(swap_image_dir)
    interval = 1
    fps = 30
    fps = video_images.video2imgs(
        input_video_path, output_frames_dir, interval=interval
    )
    faceswap_insightface.swap_faces_batch(
        output_frames_dir, target_face_path, swap_image_dir, swapper, app
    )
    output_video_path = os.path.join(
        BASE_DIR,
        target_face_path.split(".")[0].split("/")[-1]
        + "_"
        + video_name
        + "_swapped.mp4",
    )
    video_images.image2video_2(swap_image_dir, output_video_path, fps=fps / interval)
    # 删除中间目录
    shutil.rmtree(os.path.join(BASE_DIR, video_name))
    print(f"换脸后的视频保存在：{output_video_path}")
    return output_video_path


def gradio_interface(input_video, target_image):
    # 初始化FaceAnalysis和Swapper
    app = FaceAnalysis(
        name="buffalo_l",
        root="checkpoints",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": 0}, {}],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        "./checkpoints/models/inswapper_128.onnx", download=True, download_zip=True
    )

    # 进行换脸操作
    output_video_path = swapface(input_video, target_image, swapper, app)
    # 返回换脸后的视频路径
    return output_video_path


# Gradio界面
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Image(type="filepath", label="Target Face Image"),
    ],
    outputs=gr.Video(label="Swapped Video"),
    title="Video Face Swap",
    description="Upload a video and a target face image to swap faces.",
)

# 启动Gradio界面
iface.launch(server_name="0.0.0.0", debug=True)
