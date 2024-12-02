# -*-coding:utf-8 -*-

"""
# File       : video_images.py
# Time       ：2023/6/11 11:15
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import cv2
import os
import av
from tqdm import tqdm
from loguru import logger

def video2imgs(input_video_path, output_dir, interval=1):
    """

    :param ipnut_video_path:
    :param output_dir:
    :param interval:
    :return: 视频的帧率，用于视频合成
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 目标文件夹不存在，则创建
    cap = cv2.VideoCapture(input_video_path)  # 获取视频
    judge = cap.isOpened()  # 判断是否能打开成功
    print(judge)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # global fps  # 帧率，视频每秒展示多少张图片
    print("该视频的fps:", fps)

    frames = 1  # 用于统计所有帧数
    count = 1  # 用于统计保存的图片数量

    while judge:
        flag, frame = cap.read()  # 读取每一张图片 flag表示是否读取成功，frame是图片
        if not flag:
            print(flag)
            print("Process finished!")
            break
        else:
            if (frames + 1) % interval == 0:  # 每隔 interval 帧抽一张
                imgname = "pngs_" + str(count).rjust(3, "0") + ".png"
                newPath = os.path.join(output_dir, imgname)
                # print(imgname)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                count += 1
        frames += 1
    cap.release()
    print("共有 %d 张图片,%d 帧" % (count - 1, frames - 1))
    return fps


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream("h264", rate=f"{frame_rate:.4f}")
        self.stream.pix_fmt = "yuv420p"
        self.stream.bit_rate = bit_rate

    def write(self, frames):
        # frames: [ H, W,Channel] RGB
        self.stream.width = frames[0].shape[1]
        self.stream.height = frames[0].shape[0]
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            self.container.mux(self.stream.encode(frame))

    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()


def imgs2video(images_dir: str, video_saved_path: str = "./", fps: int = 30):
    image_files = os.listdir(images_dir)
    image_files = [
        image_name
        for image_name in image_files
        if image_name.split(".")[-1] in ["png", "jpg", "jpeg"]
    ]
    first_image_path = os.path.join(images_dir, image_files[0])
    img = cv2.imread(first_image_path)
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    print(size)
    filelist = image_files
    filelist.sort()
    # fps = 15   # 视频每秒组成的原始帧数，由于之前是抽帧提取图片的，我想在这里设小一点
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"MJPG")  # 设置视频编码格式
    video = cv2.VideoWriter(video_saved_path, fourcc, fps, size)
    video.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for image in filelist:
        if image.lower().split(".")[-1] in ["jpg", "png", "jpeg"]:
            image_path = os.path.join(images_dir, image)
            img = cv2.imread(image_path)
            video.write(img)

    video.release()
   


def write(img_list, size, video_saved_path: str, fps: int = 30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"MJPG")  # 设置视频编码格式
    video = cv2.VideoWriter(video_saved_path, fourcc, fps, size)
    video.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for image in img_list:
        video.write(image)  # bgr
    video.release()


def image2video_2(images_dir: str, video_saved_path: str, fps: int = 30):
    videoWriter = VideoWriter(video_saved_path, fps)
    image_files = os.listdir(images_dir)
    image_files = [
        image_name
        for image_name in image_files
        if image_name.split(".")[-1] in ["png", "jpg", "jpeg"]
    ]
    image_list = []
    for image in tqdm(sorted(image_files)):
        if image.lower().split(".")[-1] in ["jpg", "png", "jpeg"]:
            image_path = os.path.join(images_dir, image)
            img = cv2.imread(image_path)
            # videoWriter.write(img)
            image_list.append(img)

    videoWriter.write(image_list)
    videoWriter.close()


if __name__ == "__main__":
    # videoPath = "/data/bocheng/mywork/mp4_test/006.mp4"
    # imgPath = "/home/bocheng/data/mywork/style_transfer/animed_images"
    # video2imgs(videoPath, imgPath)
    # video_saved = "/home/bocheng/data/mywork/style_transfer/res1.mp4"
    img_dir = "/home/bocheng/dev/mylearn/boss.ai/MagicDance/logs/images/181020/mypose/image/0/gen_images"
    video_saved_path = "/home/bocheng/dev/mylearn/boss.ai/MagicDance/logs/images/181020/mypose/image/0/gen_images/test.mp4"
    imgs2video(images_dir=img_dir, video_saved_path=video_saved_path)
    # image2video_2(imgPath, video_saved)
