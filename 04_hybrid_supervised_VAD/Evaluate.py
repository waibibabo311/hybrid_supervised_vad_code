import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time

import pandas as pd
from tqdm import tqdm

from datasets.DataLoaderX import DataLoaderX
from datasets.UbnormalAbnormal import UbnormalDatasetAbnormal
from datasets.UbnormalNormal import UbnormalDatasetNormal
from model.pre_feature_extracter.Swin_Transformer.models.swin_transformer_v2 import SwinTransformerV2
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs for training')
parser.add_argument('--start_epoch', type=int, default=5, help='the epoch of the training starting from')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=512, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
parser.add_argument('--stage', type=str, default='validate', help='')
parser.add_argument('--t_length', type=int, default=2, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
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

# args about the feature extractor module
parser.add_argument('--pre_swin_T_checkpoint_file', type=str, default=rf'/data/gzh/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
                    help='')

args = parser.parse_args()

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.h, args.w)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_dataset_abnormal = UbnormalDatasetAbnormal(args, transforms)
test_dataset_normal = UbnormalDatasetNormal(args, transforms)
combined_dataset = dataset.ConcatDataset([test_dataset_abnormal, test_dataset_normal])
test_batch = DataLoaderX(dataset=combined_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=False, drop_last=True)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = convAE(args.c, memory_size=args.msize, feature_dim=args.fdim, key_dim=args.mdim)
pre_swin_T = SwinTransformerV2()
model.cuda()
pre_swin_T.cuda()


log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    pre_swin_T_para = torch.load(args.pre_swin_T_checkpoint_file)
    pre_swin_T.load_state_dict(pre_swin_T_para['model'])


    m_items = torch.load(os.path.join(log_dir, rf'keys{args.start_epoch}.pt'))
    m_items_test = m_items.clone()
    model_para = torch.load(os.path.join(log_dir, rf'model{args.start_epoch}.pth'))
    model.load_state_dict(model_para)


    label_list=[]
    psnr_list=[]
    feature_distance_list=[]
    scores_list=[]
    frame_path_list=[]
    model.eval()
    loop = tqdm(test_batch)
    for region_batch_list, background_batch_list, labels_batch_list, frame_path_batch in loop:
        region = region_batch_list[0]  # 抽帧训练
        background = background_batch_list[0]
        labels = labels_batch_list[0]
        print(frame_path_batch)
        # labels = label_batch_list[0]
        region = Variable(region).cuda()
        background = Variable(background).cuda()
        labels = Variable(labels).cuda()

        region = pre_swin_T(region)
        background = pre_swin_T(background)
        region_fea = torch.cat((region, background), dim=1)

        layer_norm = nn.LayerNorm([region_fea.shape[1]])
        layer_norm = layer_norm.cuda()
        region_fea = layer_norm(region_fea)

        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = \
            model.forward(region_fea, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (region_fea[0] + 1) / 2)).item() # [-1,1]-->[0,1]
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, region_fea)

        if point_sc < args.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 1)  # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)

        psnr_list.append(psnr(mse_imgs))
        feature_distance_list.append(mse_feas)
        # print("labels"+str(labels.shape))
        # print(labels.item())
        label_list.append(labels.item())
        frame_path_list.extend(frame_path_batch)

    anomaly_score_total_list=score_sum(anomaly_score_list_inv(psnr_list), anomaly_score_list(feature_distance_list),
                                       args.alpha)
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    label_list = np.asarray(label_list)

    df = pd.DataFrame({
        'Column1': label_list,
        'Column2': anomaly_score_total_list,
        'Column3': frame_path_list

    })
    df.to_excel('output.xlsx', index=False)

    accuracy = roc_auc_score(y_true=label_list, y_score=anomaly_score_total_list)
    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy*100, '%')
