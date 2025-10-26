import os
import cv2
import utils

for scene_count in range(1, 30):
    main_directory = rf"/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Origin_20231031/Scene"+str(scene_count)
    file_extension = ".mp4"
    # 遍历主目录下的所有子目录
    for root, dirs, files in os.walk(main_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_path_list = file_path.split('/')
            # 检查文件名是否以指定的文件扩展名结尾
            if filename.endswith(file_extension):
                if not os.path.exists(file_path[:-4]+"_frames"):
                    os.mkdir(file_path[:-4]+"_frames")
                cap = cv2.VideoCapture(file_path)
                frame_count = 0
                while True:
                    # 读取一帧
                    ret, frame = cap.read()

                    # 如果没有读到帧，说明视频已经结束
                    if not ret:
                        break

                    # 保存帧为图片
                    frame_path = os.path.join(file_path[:-4]+"_frames", file_path_list[-1][:-4]+"_" +
                                              str(frame_count).zfill(4)+"_frame.png")
                    cv2.imwrite(frame_path, frame)

                    frame_count += 1
                    print(frame_path + " has been written")
                    with open("output_Ubnormal_0_video2frames.txt", "a") as f:
                        print(frame_path + " has been written", file=f)

                cap.release()
with open("output_Ubnormal_1_video2frames_2.txt", "a") as f_2:
    print(" down", file=f_2)
