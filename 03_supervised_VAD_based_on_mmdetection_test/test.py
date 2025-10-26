import torch
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import numpy as np
from datasets.DataLoaderX import DataLoaderX
from datasets.UbnormalAbnormal import UbnormalDatasetAbnormal
from datasets.UbnormalNormal import UbnormalDatasetNormal

import argparse

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--start_epoch', type=int, default=0, help='the epoch of the training starting from')


parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
parser.add_argument('--stage', type=str, default='validate', help='')
parser.add_argument('--t_length', type=int, default=32, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')

parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='Ubnormal', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str,
                    default='/data/gzh/Ubnormal_For_OD',
                    help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--file_suffix', type=str, default=".mp4", help='the suffix of the loaded image filename')
parser.add_argument('--is_only_normal', type=bool, default=True,
                    help=' ')

# args about the anomaly region location
parser.add_argument('--region_location_config_file', type=str, default="/userHome/gzh/mmdetection-3.x_20240307/NewFolder/faster-rcnn_r50_fpn_2x_coco.py",
                    help='')
parser.add_argument('--region_location_checkpoint_file', type=str, default='/userHome/gzh/mmdetection-3.x_20240307/work_dirs/prompt/epoch_313.pth',
                    help=' ')

args = parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    dataset1 = UbnormalDatasetAbnormal(args)
    dataset2 = UbnormalDatasetNormal(args)

    # 连接数据集
    concat_dataset = ConcatDataset([dataset1, dataset2])

    # 创建数据加载器
    dataloader = DataLoaderX(concat_dataset, batch_size=2, shuffle=True, num_workers=args.num_workers)
    output_list=[]
    label_list=[]
    loop = tqdm(dataloader)
    for output, label in loop:
        output_list += output.tolist()
        label_list += label.tolist()
    label_list = np.asarray(label_list)
    output_list = np.asarray(output_list)

    auc = roc_auc_score(label_list, output_list)
    recall = recall_score(label_list, output_list, pos_label=1)
    accuracy = accuracy_score(label_list, output_list)

    print("Recall:", recall)
    print("Accuracy:", accuracy)
    print("AUC:", auc)