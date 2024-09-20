# -*-coding:utf-8 -*-

"""
# File       : faceswap_insightface.py
# Time       ：2023/6/12 21:48
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import datetime
from time import time

import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__ >= "0.7"
from tqdm import tqdm


def swap_faces(
    src_image_path: str,
    target_image_path: str,
    output_path: str,
    swapper: insightface.model_zoo.inswapper.INSwapper,
    app: FaceAnalysis,
):
    """
    将src照片中的脸替换为target照片中的人脸
    :param src_image_path:
    :param target_image_path:
    :param model:
    :return:
    """
    start_time = time()
    source_img = cv2.imread(src_image_path)
    target_img = cv2.imread(target_image_path)
    source_faces = app.get(source_img)  # 获取source图片中的人脸
    target_faces = app.get(target_img)  # 获取目标图片中的人脸

    source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
    target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
    if len(source_faces) == 0:
        print("no faces found in source image")

    if len(target_faces) == 0:
        print("no faces found in target image")

    res = source_img.copy()
    if len(source_faces) > 0 and len(target_faces) > 0:
        res = swapper.get(
            res, source_faces[0], target_faces[0], paste_back=True
        )  # 将source_faces中的第一张脸替换为target_faces中的第一张脸
    else:
        print("no faces found")
        res = source_img
    cv2.imwrite(output_path, res)
    end_time = time()
    print(f"结果输出到:{output_path},换脸时间为: {end_time - start_time} 秒")


def swap_faces_2(
    src_image_path: str,
    target_image_path: str,
    output_path: str,
    swapper: insightface.model_zoo.inswapper.INSwapper,
    app: FaceAnalysis,
):
    """
    将src照片中的脸替换为target照片中的人脸
    :param src_image_path:
    :param target_image_path:
    :param model:
    :return:
    """
    start_time = time()
    source_img = cv2.imread(src_image_path)
    source_faces = app.get(source_img)  # 获取source图片中的人脸
    source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
    if len(source_faces) == 0:
        print("no faces found in source image")
    if isinstance(target_image_path, str):
        target_img = cv2.imread(target_image_path)
        target_faces = app.get(target_img)  # 获取目标图片中的人脸
        target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
        if len(target_faces) == 0:
            print("no faces found in target image")
    else:
        target_faces = target_image_path

    res = source_img.copy()
    if len(source_faces) > 0 and len(target_faces) > 0:
        res = swapper.get(
            res, source_faces[0], target_faces[0], paste_back=True
        )  # 将source_faces中的第一张脸替换为target_faces中的第一张脸
    else:
        print("no faces found")
        res = source_img
    cv2.imwrite(output_path, res)
    end_time = time()
    print(f"结果输出到:{output_path},换脸时间为: {end_time - start_time} 秒")


def swap_two_faces(
    src_image_path: str,
    target_image_path_1: str,
    target_image_path_2: str,
    output_path: str,
    swapper: insightface.model_zoo.inswapper.INSwapper,
    app: FaceAnalysis,
):
    """
    将src照片中的第一和第二个人脸替换为target_image_path_1和target_image_path_2照片中的人脸
    :param src_image_path:
    :param target_image_path_1:
    :param target_image_path_2:
    :param model:
    :return:
    """
    start_time = time()
    source_img = cv2.imread(src_image_path)
    target_img_1 = cv2.imread(target_image_path_1)
    target_img_2 = cv2.imread(target_image_path_2)
    source_faces = app.get(source_img)  # 获取source图片中的人脸
    target_faces_1 = app.get(target_img_1)  # 获取目标图片中的人脸
    target_faces_2 = app.get(target_img_2)

    source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
    target_faces_1 = sorted(target_faces_1, key=lambda x: x.bbox[0])
    target_faces_2 = sorted(target_faces_2, key=lambda x: x.bbox[0])
    if len(source_faces) == 0:
        print("no faces found in source image")

    if len(target_faces_1) == 0:
        print("no faces found in target_1 image")

    if len(target_faces_2) == 0:
        print("no faces found in target_2 image")
    res = source_img.copy()
    if len(source_faces) >= 2 and len(target_faces_1) >= 1 and len(target_faces_2) >= 1:
        res = swapper.get(
            res, source_faces[0], target_faces_1[0], paste_back=True
        )  # 将source_faces中的第一张脸替换为target_faces_1中的第一张脸
        res = swapper.get(
            res, source_faces[1], target_faces_2[0], paste_back=True
        )  # 将source_faces中的第一张脸替换为target_faces_2中的第一张脸
    elif len(source_faces) >= 1 and (
        len(target_faces_1) >= 1 or len(target_faces_2) >= 1
    ):
        face = target_faces_1[0] if len(target_faces_1) > 0 else target_faces_2[0]
        res = swapper.get(
            res, source_faces[0], face, paste_back=True
        )  # 将source_faces中的第一张脸替换为target_faces_1中的第一张脸
    elif len(source_faces) < 1:
        print("no faces found in source image")
        res = source_img
        cv2.imwrite(output_path, res)
    end_time = time()
    print(f"结果输出到:{output_path},换脸时间为: {end_time - start_time} 秒")


def swap_faces_batch(
    src_images_dir: str,
    target_image_path: str,
    output_dir: str,
    swapper: insightface.model_zoo.inswapper.INSwapper,
    app: FaceAnalysis,
):
    start_time = time()
    target_img = cv2.imread(target_image_path)
    target_faces = app.get(target_img)  # 获取目标图片中的人脸
    target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
    images = os.listdir(src_images_dir)
    for image_name in tqdm(images):
        if image_name.split(".")[-1] not in ["png", "jpg"]:
            continue
        swap_faces_2(
            os.path.join(src_images_dir, image_name),
            target_faces,
            os.path.join(output_dir, "swap_" + image_name),
            swapper,
            app,
        )
    end_time = time()
    print(f"图片批量换脸总共花费时间：{end_time - start_time} 秒")


if __name__ == "__main__":
    start_time = time()
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx", download=True, download_zip=True
    )
    print(f"模型加载花费时间:{time() - start_time} 秒")

    # img = ins_get_image('t1')
    # faces = app.get(img)
    # faces = sorted(faces, key=lambda x: x.bbox[0])
    # assert len(faces) == 6
    # source_face = faces[2]
    # res = img.copy()
    # for face in faces:
    # 	res = swapper.get(res, face, source_face, paste_back=True)
    # cv2.imwrite("./t1_swapped.jpg", res)
    # res = []
    # for face in faces:
    # 	_img, _ = swapper.get(img, face, source_face, paste_back=False)
    # 	res.append(_img)
    # res = np.concatenate(res, axis=1)
    # cv2.imwrite("./t1_swapped2.jpg", res)

    target_image_path = "/home/bocheng/dev/python/ai-imagegen/data/face_images/qiqi.jpg"
    src_image_path = "/home/bocheng/dev/python/ai-imagegen/data/face_images/ergou.jpg"
    output_path = "./result_swap.jpg"
    swap_faces(src_image_path, target_image_path, output_path, swapper, app)
    # print(f"总共花费时间：{time() - start_time}")
    pass
