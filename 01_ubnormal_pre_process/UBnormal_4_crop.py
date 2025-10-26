import os
import cv2
import utils
import numpy as np


for scene_count in range(1, 30):
    main_directory = rf"/userHome/ljx/guozhihai/UBnormal_Origin_20231031/Scene" + str(scene_count)
    file_extension = ".mp4"
    # 遍历主目录下的所有子目录
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_path_list = file_path.split('/')
            # 检查文件名是否以指定的文件扩展名结尾
            if filename.endswith(file_extension):
                cap = cv2.VideoCapture(file_path)
                region_count = 0
                if not os.path.exists(file_path[:-4]+"_regions"):
                    os.mkdir(file_path[:-4]+"_regions")
                if not os.path.exists(file_path[:-4]+"_backgrounds"):
                    os.mkdir(file_path[:-4]+"_backgrounds")
                while True:
                    # 逐帧读取视频
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotations_path = file_path[:-4] + "_annotations"
                    # current_frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    frame_number = str(region_count).zfill(4) + "_gt.png"
                    mask_path = os.path.join(annotations_path, file_path_list[-1][:-4] + "_" + frame_number)
                    # print(mask_path)
                    if not os.path.exists(mask_path):
                        continue
                    if not ("abnormal" in mask_path):
                        continue
                    mask = cv2.imread(mask_path)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    background = frame.copy()
                    # print(region_count)
                    rectangles = utils.calculate_rectangles(mask)
                    for i, (x, y, w, h) in enumerate(rectangles):
                        background[y:y + h, x:x + w] = 0

                        background_path = os.path.join(file_path[:-4] + "_backgrounds", file_path_list[-1][:-4] + "_" +
                                                       str(region_count).zfill(4) + "_background.png")
                        cv2.imwrite(background_path, background)
                        crop = frame[y:y + h, x:x + w]
                        region_path = os.path.join(file_path[:-4]+"_regions", file_path_list[-1][:-4] + "_" +
                                                   str(region_count).zfill(4) + "_region.png")
                        # frame_path = os.path.join(rf"/userHOME/ljx/guozhihai/UBnormal/Regions",
                        # str(region_count).zfill(6) + "_frame.png")
                        cv2.imwrite(region_path, crop)
                        # cv2.imwrite(frame_path, frame)
                        print(mask_path + " has been cropped")
                        with open("output_UBnormal_4_crop.txt", "a") as f:
                            print(mask_path + " has been cropped", file=f)
                    region_count += 1
                cap.release()
with open("output_UBnormal_3_crop_2.txt.txt", "a") as f_2:
    print("down", file=f_2)
