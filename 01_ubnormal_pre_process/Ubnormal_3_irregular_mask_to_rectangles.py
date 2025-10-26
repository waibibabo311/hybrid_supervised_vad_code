import numpy as np
from utils import irregular_mask_to_rectangles
from utils import merge_rectangles
from utils import calculate_rectangles
from utils import insert_char_after_char
import os
import cv2
import datetime

file_extension = "_gt.png"

for scene_count in range(1, 2):
    main_directory = rf"/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Origin_20231031/Scene"+str(scene_count)
    file_extension = "_gt.png"
    # 遍历主目录下的所有子目录
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            # 使用 os.path.join() 函数获取完整的文件路径
            file_path = os.path.join(root, filename)

            # 检查文件名是否以指定的文件扩展名结尾
            if filename.endswith(file_extension):
                img = cv2.imread(file_path)
                region_file_path = file_path
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                height, width = img.shape

                img = np.array(img)
                img = irregular_mask_to_rectangles(img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rois = calculate_rectangles(img)
                # print(len(rois))
                if len(rois) > 1:
                    rois = merge_rectangles(rois)
                for i, (x, y, w, h) in enumerate(rois):
                    img[y:y + h, x:x + w] = 255
                    cv2.imwrite(region_file_path, img)
                    print(str(datetime.datetime.now())+"已写入" + region_file_path)
                with open("output_Ubnormal_3_irregular_mask_to_rectangles.txt", "a") as f:
                    print(str(datetime.datetime.now())+"已写入"+region_file_path, file=f)
with open("Ubnormal_2_irregular_mask_to_rectangles_2.txt", "a") as f_2:
    print("done", file=f_2)
