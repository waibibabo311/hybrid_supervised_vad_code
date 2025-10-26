import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import datetime

for scene_count in range(1, 30):
    main_directory = rf"/data/gzh/Ubnormal_For_OD/Scene"+str(scene_count)
    png_file_extension = "_gt.png"

    # only the anomaly object pixels are 255, the other pixels are all 0
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            if filename.endswith(png_file_extension):
                file_path = os.path.join(root, filename)
                img = Image.open(file_path)
                img = np.array(img)
                img[img != 255] = 0
                img = Image.fromarray(img)
                img.save(file_path)
                print(str(datetime.datetime.now()) + "已写入" + file_path)
                with open("output_Ubnormal_1_cvtColor.txt", "a") as f:
                    print(str(datetime.datetime.now()) + "已写入" + file_path, file=f)
with open("output_Ubnormal_2_HighLight_Objects_2.txt", "a") as f_2:
    print("done", file=f_2)