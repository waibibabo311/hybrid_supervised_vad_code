import os
import numpy as np
from PIL import Image
import math  # 导入math模块

for scene_count in range(1, 2):
    main_directory = rf"/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Origin_20231031/Scene" + str(scene_count)
    file_extension = "_gt.png"
    # 遍历主目录下的所有子目录
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            # 检查文件名是否以指定的文件扩展名结尾
            if filename.endswith(file_extension):
                input_image = Image.open(file_path)
                # 计算调整大小后的图像尺寸，保持原始纵横比

                if input_image.mode == 'RGB':
                    input_image = input_image.convert("L")


                width, height = input_image.size
                if width > height:
                    new_width = 1024
                    new_height = int(math.floor(height * (1024 / width)))  # 向下取整
                else:
                    new_width = int(math.floor(width * (1024 / height)))  # 向下取整
                    new_height = 1024

                # 调整大小并将图像转换为NumPy数组
                resized_image = input_image.resize((new_width, new_height), Image.ANTIALIAS)
                resized_array = np.array(resized_image)

                # 创建一个1024x1024的零矩阵
                output_array = np.zeros((1024, 1024), dtype=np.uint8)

                # 计算将图像置于中心时的位置
                x_offset = (1024 - new_width) // 2
                y_offset = (1024 - new_height) // 2

                # 将调整大小后的图像粘贴到零矩阵上
                output_array[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_array

                # 将NumPy数组转换为图像
                output_image = Image.fromarray(output_array)

                # 保存结果
                output_image.save(file_path)

                print(file_path + " has been written")
                with open("output_Ubnormal_8_gt_resize.txt", "a") as f:
                    print(file_path[:-4]+"_resize.png" + " has been written", file=f)

with open("output_Ubnormal_8_gt_resize.txt", "a") as f_2:
    print("down", file=f_2)
