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

from tqdm import tqdm

from model.pre_feature_extracter.Swin_Transformer.models.swin_transformer_v2 import SwinTransformerV2
from model.utils import DataLoader
from sklearn.metrics import roc_auc_score
from datasets.DataLoaderX import DataLoaderX
from datasets.UbnormalNormal import UbnormalDatasetNormal
from utils import *
import random

import argparse

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--start_epoch', type=int, default=6, help='the epoch of the training starting from')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=512, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
parser.add_argument('--stage', type=str, default='train', help='')
parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=256, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=256, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=1000, help='number of the memory items')
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset_normal = UbnormalDatasetNormal(args, transforms)

    train_batch = DataLoaderX(dataset=train_dataset_normal, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)

    # Model setting
    assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
    if args.method == 'pred':
        from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *

        model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    else:
        from model.Reconstruction import *


        model = convAE(args.c, memory_size=args.msize, feature_dim=args.fdim, key_dim=args.mdim)
        pre_swin_T = SwinTransformerV2()
    params_encoder = list(model.encoder.parameters())
    params_decoder = list(model.decoder.parameters())
    params = params_encoder + params_decoder
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()
    pre_swin_T.cuda()


    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    loss_func_mse = nn.MSELoss(reduction='none')

    # Training

    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float),
                          dim=1).cuda()  # Initialize the memory items

    for epoch in range(args.start_epoch, args.epochs):
        pre_swin_T_para = torch.load(args.pre_swin_T_checkpoint_file)
        pre_swin_T.load_state_dict(pre_swin_T_para['model'])

        if epoch > 0:
            m_items = torch.load(os.path.join(log_dir, rf'keys{epoch}.pt'))
            model_para = torch.load(os.path.join(log_dir, rf'model{epoch}.pth'))
            model.load_state_dict(model_para)
        labels_list = []
        model.train()

        start = time.time()
        loop = tqdm(train_batch)
        for region_batch_list, background_batch_list, _ in loop:
            region = region_batch_list[0]  # 抽帧训练

            background = background_batch_list[0]

            # labels = label_batch_list[0]
            region = Variable(region).cuda()
            background = Variable(background).cuda()

            if args.method == 'pred':
                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(
                    region[:, 0:12], m_items, True)

            else:
                region = pre_swin_T(region)
                background = pre_swin_T(background)
                region_fea = torch.cat((region, background), dim=1)

                layer_norm = nn.LayerNorm([region_fea.shape[1]])
                layer_norm = layer_norm.cuda()
                region_fea = layer_norm(region_fea)

                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(
                    region_fea, m_items, True)

            optimizer.zero_grad()
            if args.method == 'pred':
                loss_pixel = torch.mean(loss_func_mse(outputs, region_fea[:, 12:]))
            else:
                loss_pixel = torch.mean(loss_func_mse(outputs, region_fea))

            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            print('Epoch:', epoch + 1)
            loop.set_postfix(loss=rf"{loss:.6f}")
            loss.backward(retain_graph=True)
            optimizer.step()

        scheduler.step()

        print('----------------------------------------')
        print('Epoch:', epoch + 1)
        if args.method == 'pred':
            print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(),
                                                                                            compactness_loss.item(),
                                                                                            separateness_loss.item()))
        else:
            print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(),
                                                                                                compactness_loss.item(),
                                                                                                separateness_loss.item()))
        print('Memory_items:')
        print(m_items)
        print('----------------------------------------')
        # Save the model and the memory items
        torch.save(model.state_dict(), os.path.join(log_dir, rf'model{epoch + 1}.pth'))
        torch.save(m_items, os.path.join(log_dir, rf'keys{epoch + 1}.pt'))

    print('Training is finished')

