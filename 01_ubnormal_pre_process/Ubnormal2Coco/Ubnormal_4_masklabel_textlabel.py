import os

import cv2
import numpy as np
from PIL import Image
import math  # 导入math模块

import utils

for scene_count in range(1, 30):
    main_directory = rf"/data/gzh/Ubnormal_For_OD/Scene" + str(scene_count)
    file_extension = "_gt.png"
    # 遍历主目录下的所有子目录
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            # 检查文件名是否以指定的文件扩展名结尾
            if filename.endswith(file_extension):

                mask = cv2.imread(file_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                rectangles = utils.calculate_rectangles(mask)
                with open(file_path[:-3]+'txt', 'w'):
                    pass  # 不执行任何写入操作
                for i, (x, y, w, h) in enumerate(rectangles):
                    print(h)
                    print('--------------')
                    text = f"{x},{y},{w},{h},1,1,0,0\n"

                    # 将文本写入文件
                    with open(file_path[:-3]+'txt', 'w') as file:

                        file.write(text)
                print(file_path[:-3]+'txt' + " has been written")
                with open("Ubnormal_9_masklabel_textlabel", "a") as f:
                    print(file_path[:-3]+'txt' + " has been written", file=f)

with open("Ubnormal_9_masklabel_textlabel.txt", "a") as f_2:
    print("down", file=f_2)
