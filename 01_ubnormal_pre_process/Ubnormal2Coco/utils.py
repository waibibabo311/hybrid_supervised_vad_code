import cv2
import os
import numpy as np
from scipy import ndimage

def extract_frames(video_path, output_path):
    # open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("无法打开视频")
        return
    # create the output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # read the video and store every frame
    frame_count = 0
    while True:
        # read every frame
        ret, frame = video.read()
        if not ret:
            break
        # the filename of the output
        output_filename = os.path.join(output_path, str(frame_count).zfill(5) + "_frame.png")
        # store frame
        cv2.imwrite(output_filename, frame)
        frame_count += 1
    video.release()


def change_frame_rate(input_path, output_path, target_frame_rate):
    # 打开输入视频文件
    video = cv2.VideoCapture(input_path)

    # 获取输入视频的帧率和总帧数
    input_frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算每一帧的时间间隔
    input_frame_interval = 1 / input_frame_rate
    target_frame_interval = 1 / target_frame_rate

    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码方式
    video_writer = cv2.VideoWriter(output_path, fourcc, target_frame_rate, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # 初始化计数器和时间戳
    input_frame_count = 0
    target_frame_count = 0
    target_timestamp = 0

    # 逐帧读取和写入视频
    while input_frame_count < total_frames:
        success, frame = video.read()
        if not success:
            break

        # 计算当前帧的时间戳
        input_timestamp = input_frame_count * input_frame_interval

        # 判断是否需要写入当前帧
        if input_timestamp >= target_timestamp:
            video_writer.write(frame)
            target_frame_count += 1
            target_timestamp += target_frame_interval

        input_frame_count += 1

    # 释放视频文件和视频编写器
    video.release()
    video_writer.release()

    print(f"已成功改变帧率，新视频共有 {target_frame_count} 帧.")


def calculate_rectangles(image):

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算轮廓的边界框
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 由元组组成的列表
        rectangles.append((x, y, w, h))

    return rectangles


def count_files(directory):
    file_count = 0

    # 遍历目录中的文件和文件夹
    for _, _, files in os.walk(directory):
        file_count += len(files)

    return file_count

def merge_rectangles(rectangles):
    # rectangles是一个包含多个矩形的列表，每个矩形由(x, y, width, height)表示
    if not rectangles:
        return None

    # 计算合并后的矩形的左上角和右下角坐标
    x_min = min(rect[0] for rect in rectangles)
    y_min = min(rect[1] for rect in rectangles)
    x_max = max(rect[0] + rect[2] for rect in rectangles)
    y_max = max(rect[1] + rect[3] for rect in rectangles)

    merge_rectangle = [(x_min, y_min, x_max - x_min, y_max - y_min)]

    return merge_rectangle

def irregular_mask_to_rectangles(irregular_mask):
    labeled_mask, num_regions = ndimage.label(irregular_mask)
    rectangles = []
    # print(num_regions)
    rectangle_mask = np.zeros_like(irregular_mask)
    for region_label in range(1, num_regions + 1):
        region_pixels = np.argwhere(labeled_mask == region_label)
        min_y, min_x = np.min(region_pixels, axis=0)
        max_y, max_x = np.max(region_pixels, axis=0)


        rectangle_mask[min_y:max_y + 1, min_x:max_x + 1] = 255

    return rectangle_mask


def insert_char_after_char(original_string, char_to_insert, target_char):
    try:
        position = original_string.index(target_char) + len(target_char)
        new_string = original_string[:position] + char_to_insert + original_string[position:]
        return new_string
    except ValueError:
        # 如果目标字符不在原始字符串中，直接返回原始字符串
        return original_string