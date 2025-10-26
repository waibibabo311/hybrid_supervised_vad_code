import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import datetime

for scene_count in range(1, 2):
    main_directory = rf"/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Origin_20231031/Scene"+str(scene_count)
    txt_file_extension = "_tracks.txt"
    png_file_extension = ".gt_png"

    # Highlight the anomaly objects
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            if filename.endswith(txt_file_extension):
                # 使用 os.path.join() 函数获取完整的文件路径
                file_path = os.path.join(root, filename)
                file_path_list = file_path.split('/')
                print("\n"+file_path)
                tracks = np.loadtxt(file_path, delimiter=",")
                print(tracks)
                if len(tracks.shape) == 1:
                    tracks = tracks[np.newaxis, :]
                for track_index, track in enumerate(tracks):
                    print(track)
                    object_id = int(track[0])
                    object_start_frame = int(track[1])
                    object_end_frame = int(track[2])

                    for i in range(object_start_frame, object_end_frame + 1):
                        img_path = os.path.join(root, file_path_list[-1][:-10] + str(i).zfill(4)) + "_gt.png"
                        img = Image.open(img_path)
                        img = np.array(img)
                        img[img == object_id] = 255
                        img = Image.fromarray(img)
                        img.save(img_path)
                        print(str(datetime.datetime.now())+"已写入" + img_path)
                        with open("output_Ubnormal_1_cvtColor.txt", "a") as f:
                            print(str(datetime.datetime.now())+"已写入" + file_path, file=f)



with open("output_Ubnormal_1_cvtColor_2.txt", "a") as f_2:
    print("done", file=f_2)
