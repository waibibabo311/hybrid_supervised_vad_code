from collections import OrderedDict
import cv2
import glob
from PIL import Image
import torch
from model.pre_feature_extracter.utils import anomaly_region_location, check_txt_file
from torch.utils.data import Dataset, DataLoader
import json
from mmdet.apis import init_detector, inference_detector
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np


class UbnormalDatasetAbnormal(Dataset):
    def __init__(self, args, transform):
        self.root_dir = args.dataset_path
        self.transform = transform
        self.videos = OrderedDict()
        self.filename_suffix = args.file_suffix
        self.time_step = args.t_length
        self.stage = args.stage
        self.is_only_normal = args.is_only_normal
        self.region_location_model = \
            init_detector(config=args.region_location_config_file, checkpoint=args.region_location_checkpoint_file,
                          device='cuda:0')
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(self.filename_suffix):
                    video_path = os.path.join(dirpath, filename)
                    video_paths.append(video_path)
        for video_path in sorted(video_paths):
            video_name = video_path.split('/')[-1][:-4]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video_path
            # a list
            all_frames = glob.glob(os.path.join(video_path[:-4] + "_frames", '*_frame.png'))
            all_frames.sort()
            self.videos[video_name]['key_frame'] = all_frames[:-self.time_step:self.time_step] # 保证每个key_frame都有后继
            self.videos[video_name]['length'] = len(self.videos[video_name]['key_frame'])
            # print(self.videos[video_name]['length'])

        # print(self.videos)

    def get_all_samples(self):
        frame_paths = []
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(self.filename_suffix):
                    video_path = os.path.join(dirpath, filename)
                    video_paths.append(video_path)

        if self.stage == "train":
            with open(rf"./datasets/scripts/abnormal_training_video_names.txt", 'r') as file:
                for line in file:
                    video_name = line[:-1]
                    # print("self.videos", self.videos)
                    for i in range(len(self.videos[video_name]['key_frame'])):
                        frame_paths.append(self.videos[video_name]['key_frame'][i])
            # with open(rf"./datasets/scripts/normal_training_video_names.txt", 'r') as file:
            #     for line in file:
            #         video_name = line[:-1]
            #         for i in range(len(self.videos[video_name]['key_frame'])):
            #             frame_paths.append(self.videos[video_name]['key_frame'][i])

        if self.stage == "validate":
            with open(rf"./datasets/scripts/abnormal_validation_video_names.txt", 'r') as file:
                for line in file:
                    video_name = line[:-1]
                    for i in range(len(self.videos[video_name]['key_frame'])):
                        frame_paths.append(self.videos[video_name]['key_frame'][i])
            # with open(rf"./datasets/scripts/normal_validation_video_names.txt", 'r') as file:
            #     for line in file:
            #         video_name = line[:-1]
            #         for i in range(len(self.videos[video_name]['key_frame'])):
            #             frame_paths.append(self.videos[video_name]['key_frame'][i])

        if self.stage == "test":
            with open(rf"./datasets/scripts/abnormal_test_video_names.txt", 'r') as file:
                for line in file:
                    video_name = line[:-1]
                    for i in range(len(self.videos[video_name]['key_frame'])):
                        frame_paths.append(self.videos[video_name]['key_frame'][i])
            # with open(rf"./datasets/scripts/normal_test_video_names.txt", 'r') as file:
            #     for line in file:
            #         video_name = line[:-1]
            #         for i in range(len(self.videos[video_name]['key_frame'])):
            #             frame_paths.append(self.videos[video_name]['key_frame'][i])

        return frame_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print(idx)
        # video_name = self.samples[idx].split('/')[-2][:-7]  # -7 _frames   sample_id --> frame_path --> video_name
        # 10 _frame.png sample_id --> frame_path --> frame_number
        frame_number = int(self.samples[idx].split('/')[-1][-14:-10])
        # print(frame_number)
        batch_region = []
        batch_background = []
        batch_label = []
        for i in range(self.time_step):
            frame_path = self.samples[idx][:-14] + str(frame_number + i).zfill(4) + self.samples[idx][-10:]

            gt_path = frame_path.replace("_frames", "_annotations").replace("_frame.png", "_gt.txt")
            label = check_txt_file(gt_path)
            batch_label.append(label)

            region, background = anomaly_region_location(self.region_location_model, frame_path)

            if self.transform is not None:
                region = self.transform(region)
                background = self.transform(background)

            batch_region.append(region)
            batch_background.append(background)

        return batch_region, batch_background, batch_label, frame_path