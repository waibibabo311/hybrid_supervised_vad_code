import shutil
from collections import OrderedDict
import glob
import os
from torch.utils.data import Dataset


class UbnormalCopyCoco(Dataset):

    def __init__(self, root_dir, dest_dir, filename_suffix, time_step, stage):
        self.root_dir = root_dir
        self.dest_dir = dest_dir
        self.videos = OrderedDict()
        self.filename_suffix = filename_suffix
        self.time_step = time_step
        self.stage = stage
        self.setup()
        self.copy_all_samples()

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

    def copy_all_samples(self):
        if self.stage == "train":
            self.dest_dir=os.path.join(self.dest_dir, 'train')
            if not os.path.exists( self.dest_dir):
                os.mkdir( self.dest_dir)
            with open(rf"./datasets/scripts/normal_training_video_names.txt", 'r') as file:
                for line in file:
                    video_name = line[:-1]
                    for i in range(len(self.videos[video_name]['key_frame'])):
                        frame_path=self.videos[video_name]['key_frame'][i]
                        gt_path=frame_path.replace('_frame.png', '_gt.txt').replace('_frames', '_annotations')
                        if not os.path.exists(gt_path):
                            with open(gt_path, 'w') as file:
                                pass
                        dest_frame_path = os.path.join(self.dest_dir, 'images')
                        if not os.path.exists(dest_frame_path):
                            os.mkdir(dest_frame_path)
                        dest_gt_path = os.path.join(self.dest_dir, 'annotations')
                        if not os.path.exists(dest_gt_path):
                            os.mkdir(dest_gt_path)
                        shutil.copy2(frame_path, os.path.join(dest_frame_path,
                                                              frame_path.split('/')[-1].replace('_frame.png', '.jpg')))

                        shutil.copy2(gt_path,
                                     os.path.join(dest_gt_path, gt_path.split('/')[-1].replace('_gt.txt', '.txt')))
                        print(gt_path)

        if self.stage == "validate":
            self.dest_dir = os.path.join(self.dest_dir, 'validate')
            if not os.path.exists( self.dest_dir):
                os.mkdir( self.dest_dir)
            with open(rf"./datasets/scripts/normal_validation_video_names.txt", 'r') as file:
                for line in file:
                    video_name = line[:-1]
                    for i in range(len(self.videos[video_name]['key_frame'])):
                        frame_path = self.videos[video_name]['key_frame'][i]
                        gt_path = frame_path.replace('_frame.png', '_gt.txt').replace('_frames', '_annotations')
                        if not os.path.exists(gt_path):
                            with open(gt_path, 'w') as file:
                                pass
                        dest_frame_path = os.path.join(self.dest_dir, 'images')
                        if not os.path.exists(dest_frame_path) :
                            os.mkdir(dest_frame_path)
                        dest_gt_path = os.path.join(self.dest_dir, 'annotations')
                        if not os.path.exists(dest_gt_path):
                            os.mkdir(dest_gt_path)
                        shutil.copy2(frame_path, os.path.join(dest_frame_path,
                                                              frame_path.split('/')[-1].replace('_frame.png', '.jpg')))
                        shutil.copy2(gt_path,
                                     os.path.join(dest_gt_path, gt_path.split('/')[-1].replace('_gt.txt', '.txt')))
                        print(gt_path)

        if self.stage == "test":
            self.dest_dir = os.path.join(self.dest_dir, 'test')
            if not os.path.exists( self.dest_dir):
                os.mkdir( self.dest_dir)
            with open(rf"./datasets/scripts/normal_test_video_names.txt", 'r') as file:
                for line in file:
                    video_name = line[:-1]
                    for i in range(len(self.videos[video_name]['key_frame'])):
                        frame_path = self.videos[video_name]['key_frame'][i]
                        gt_path = frame_path.replace('_frame.png', '_gt.txt').replace('_frames', '_annotations')
                        if not os.path.exists(gt_path):
                            with open(gt_path, 'w') as file:
                                pass
                        dest_frame_path = os.path.join(self.dest_dir, 'images')
                        if not os.path.exists(dest_frame_path):
                            os.mkdir(dest_frame_path)
                        dest_gt_path = os.path.join(self.dest_dir, 'annotations')
                        if not os.path.exists(dest_gt_path):
                            os.mkdir(dest_gt_path)
                        shutil.copy2(frame_path, os.path.join(dest_frame_path,
                                                              frame_path.split('/')[-1].replace('_frame.png', '.jpg')))
                        shutil.copy2(gt_path,
                                     os.path.join(dest_gt_path, gt_path.split('/')[-1].replace('_gt.txt', '.txt')))
                        print(gt_path)

if __name__ == '__main__':
    train_copy = UbnormalCopyCoco(root_dir='/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Origin_20231031', dest_dir='/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Coco_20240229', filename_suffix='.mp4', time_step=8, stage='train')
    validate_copy = UbnormalCopyCoco(root_dir='/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Origin_20231031', dest_dir='/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Coco_20240229', filename_suffix='.mp4', time_step=8, stage='validate')
    test_copy = UbnormalCopyCoco(root_dir='/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Origin_20231031', dest_dir='/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Coco_20240229', filename_suffix='.mp4', time_step=8, stage='test')
