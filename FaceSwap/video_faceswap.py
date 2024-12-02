# -*-coding:utf-8 -*-

"""
# File       : faceswap_video.py
# Time       ：2023/6/13 12:24
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import sys

sys.path.append("..")
import os.path

import faceswap_insightface
import video_images
import insightface
from insightface.app import FaceAnalysis


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
    :param output_frames_dir:输出帧的目录
    :param target_face_path:包含目标face的图片
    :param swap_image_dir:换脸后保存图片的目录
    :return:
    """
    base_dir = os.path.join(
        os.path.dirname(input_video_path),
        os.path.splitext(os.path.basename(input_video_path))[0],
    )
    output_frames_dir = os.path.join(base_dir, "output_frames")
    swap_image_dir = os.path.join(base_dir, "swapped_frames")
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
    os.path.splitext(input_video_path)
    video_images.imgs2video(
        swap_image_dir,
        os.path.join(
            base_dir,
            target_face_path.split(".")[0].split("/")[-1] + "_" + "swapped.mp4",
        ),
        fps=fps / interval,
    )


if __name__ == "__main__":
    input_video_path = "/data/bocheng/dev/mylearn/ai-imagegen/face/input/lilisi-018.mp4"
    if not os.path.exists(input_video_path):
        print("输入视频文件不存在！")
        raise Exception("Input video path does not exist")
    # output_frames_dir = os.path.join(dir, "output_frames")
    # swap_image_dir = os.path.join(dir, "swap_images")
    target_image_path = "/data/bocheng/dev/mylearn/ai-imagegen/face/input/个人照片.jpg"
    if not os.path.exists(target_image_path):
        print("Target image does not exist")
        raise Exception("Target image path does not exist")
    app = FaceAnalysis(
        name="buffalo_l",
        root="checkpoints",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": 0}],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        "./checkpoints/models/inswapper_128.onnx", download=True, download_zip=True
    )
    swapface(
        input_video_path,
        target_face_path=target_image_path,
        swapper=swapper,
        app=app,
    )
