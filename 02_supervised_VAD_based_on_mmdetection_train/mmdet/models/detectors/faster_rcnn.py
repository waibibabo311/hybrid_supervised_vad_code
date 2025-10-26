# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
import cv2

@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)


# 32-622行
# # Copyright (c) OpenMMLab. All rights reserved.
# from mmdet.registry import MODELS
# from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmdet.structures import SampleList
# from mmdet.models.necks import FPN
# from mmdet.registry import MODELS, TASK_UTILS
# from .two_stage import TwoStageDetector
# # from .mycode import DimensionReductionMLP
# from ..utils import (filter_scores_and_topk, select_single_mlvl,
#                      unpack_gt_instances)
# import copy
# import torch
# from torch import Tensor, nn
# import sys
# from transformers import BertTokenizer, BertModel,BertConfig
# import numpy as np
# from mmdet.models.detectors.conv1 import FR
# from transformers import logging
# import torch.nn.functional as F
# import os
# from transformers import CLIPProcessor, CLIPModel
# import clip
# device = torch.device('cuda')
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# #获取文本特征
# def get_text_feature(prompt): 
#     # Load the pre-trained BERT model and tokenizer
#     # 指定模型文件夹的路径
#     #prompt是个列表 里面包含多一个batch中张图像的prompt
#     model_path = "/userHome/yym/code/faster_rcnn/modeling_bert"
#     logging.set_verbosity_error()#去除警告

#     tokenizer = BertTokenizer.from_pretrained('/userHome/yym/code/faster_rcnn/modeling_bert')
#     model = BertModel.from_pretrained('/userHome/yym/code/faster_rcnn/modeling_bert',)
#     # sentenceA = 'a small people is partially truncated in the top left of photo,a small people is heavily occluded in the left of photo, a small pedestrain is partially truncated in the top left of photo, There is a dense area in the right of photo, which contain nine pedestrains,one people'
#     sentence_features = []
#     for p in prompt:
       
#         sentences = p.split('\n')
#         for text in sentences:
#             if text.strip():
#                 text_dict = tokenizer.encode_plus(text, add_special_tokens=True, return_attention_mask=True, truncation=True, max_length=512)
#                 input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
#                 token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
#                 attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)
#                 res = model(input_ids, attention_mask=attention_mask)
#                 res = res.pooler_output
#                 text_descriptions_features = res.squeeze(0)
#                 sentence_features.append(text_descriptions_features)

#     if not sentence_features:  # 如果所有的sentence_features都为空，则返回一个空的特征张量
#         return torch.empty((0, 768))
#     features_tensor=torch.stack(sentence_features, dim=0)  

#     return features_tensor

# def get_text_feature_CLIP(prompts,object):
#     # # 定义本地目录来存储预训练模型权重
#     # model_dir = "/userHome/yym/code/mmdetection-3.x/CLIP"
#     # logging.set_verbosity_error()#去除警告

#     # # 加载本地预训练模型权重
#     # processor = CLIPProcessor.from_pretrained(model_dir)
#     # model = CLIPModel.from_pretrained(model_dir)
#     # # converted_list = []
#     #  # 加载本地预训练模型权重
#     # text_features=[]
#     # processor = CLIPProcessor.from_pretrained(model_dir)
#     # model = CLIPModel.from_pretrained(model_dir)
#     # for prompt in prompts:
#     #     for text in prompt:
#     #         # print(text)
#     #         if not prompt:
#     #             text_features.append(torch.tensor([]))
#     #         else:
#     #             inputs = processor(text, return_tensors="pt", padding=True, truncation=True)
#     #             text_feature = model.get_text_features(**inputs)
#     #             text_features.append(text_feature)
#     # return torch.cat(text_features,dim=0)
#     model, processor= clip.load("ViT-B/32", device=device)
#     # Freeze the parameters of the text encoder
#     # for param in model.text_model.parameters():
#     #     param.requires_grad = False
#     text_features=[]
#     for prompt in prompts:
#         if not prompt:
#                 text_features.append(torch.tensor([]))
#                 continue
#         for text in prompt:
#             text_input=clip.tokenize(text).to(device)
#             text_feature = model.encode_text(text_input)
#             # print(text_feature.shape)
#             text_features.append(text_feature)     
#     return  text_features

# class GlobalAvgPool2d(nn.Module):
#             def forward(self, x):
#         # 进行全局平均池化操作
#                 return nn.functional.adaptive_avg_pool2d(x.unsqueeze(0), (1, 1)).squeeze(-1).squeeze(-1)

# output_size = 512
# global_avg_pool = GlobalAvgPool2d()
# global_avg_pool.to(device)
# mlp = nn.Linear(256, output_size)
# mlp.to(device)
# def extract_dense_region_features(upscaled_features, dense_region,img_path,sizes_img,flip_directions):
#                region_features_batch = []
#             #    batch_size = upscaled_features.shape[0]  # 获取批次大小
#                sizes = []
#                # print(batch_size)
#                for batch_feature, batch_dense_region,size_img,flip_direction in zip(upscaled_features, dense_region,sizes_img,flip_directions):
                   
#                    # region_features_batch = []  # 存储每个batch中的region_features
#                 #    from PIL import Image, ImageDraw
#                 #    # 从图像路径中提取文件名
#                 #    image_name = os.path.basename(image)
#                 #    image = Image.open(image)
#                 #    # 创建一个绘图对象
#                 #    draw = ImageDraw.Draw(image)
#                 #    print("GT密集区域",batch_dense_region)
#                    batch_feature=batch_feature.unsqueeze(0)
#                 #    print("backbone特征尺寸",batch_feature.shape)
#                    size_img = (size_img[0], size_img[1])
#                    mid=size_img[1]//2
#                 #    print("宽",size_img[1])
#                 #    print("高",size_img[0])
#                    batch_feature=F.interpolate(batch_feature, size=size_img, mode='bilinear', align_corners=True)
#                 #    print("原图尺寸",batch_feature.shape)
#                    for image_regions in batch_dense_region:
#                     #    print("密集区域",image_regions)
#                        x_min, y_min, width, height = image_regions               
#                        x_max = x_min + width
#                        y_max = y_min + height
                       
#                        if flip_direction=="horizontal":
#                            x_min=2*mid-x_min
#                            x_max=2*mid-x_max
#                            region_feature = batch_feature[:,:, y_min:y_max, x_max:x_min].clone()
#                         #    print(x_min, y_min,x_max,y_max)   
#                        else:
#                            region_feature = batch_feature[:,:, y_min:y_max, x_min:x_max].clone()
                       
#                        # print(batch_feature)
#                        #加载图像
                       
#                        region_features_batch.append(region_feature)
#                        sizes.append(region_feature.size())
#                     #    print("密集区域特征尺寸",region_feature.size())
#                     #    print("密集区域原始尺寸",region_feature.shape)
#                 #        outline_color = (0, 255, 0)  # 绿色边界框
#                 #        draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color)
#                 #     # 保存图像到指定文件夹
#                 #    output_folder = "/userHome/yym/code/mmdetection-3.x/output_image"  # 替换为实际的输出文件夹路径
                   

#                 #    output_path = os.path.join(output_folder,image_name)
#                 #    image.save(output_path)

                       
#                     # region_feature=global_avg_pool(region_feature)
#                     # region_feature=mlp(region_feature.squeeze())
#                     # print("密集区域降维后的特征维度：", region_feature.shape)
#                     # # 如果 region_feature 为空，跳过当前region的处理
#                     # if region_feature.numel() == 0:
#                     #     continue
#                     # image_features.append(region_feature)
        
#                 # # 将当前batch中的所有图像的密集区域特征堆叠成一个张量，并添加到结果列表中
#                 # if len(region_features_batch) > 0:
#                 #     region_features_batch = torch.stack(region_features_batch, dim=0)
#                 #     image_features.append(region_features_batch)
#              # 如果 image_features 列表为空，则返回一个空的特征张量
#             # print(len(image_features))
#             # 找到最大的切片大小
#                max_height = max(size[3] for size in sizes)
#                max_width = max(size[2] for size in sizes)
#             #    print(max_width,max_height)
#                # 填充小于最大大小的切片
#                for i in range(len(region_features_batch)):
#                    pad_height = max_height - region_features_batch[i].size(3)
#                    pad_width = max_width - region_features_batch[i].size(2)
#                 #    print("需要填充的宽高",pad_width,pad_height)
#                 #    print("填充前的维度",region_features_batch[i].shape)
#                    region_features_batch[i] = nn.functional.pad(region_features_batch[i], (0,  pad_height, 0,pad_width))
#                 #    print("填充后的维度",region_features_batch[i].shape)
#                if len(region_features_batch) > 0:
#                    # 在需要释放GPU内存的地方调用
#                    torch.cuda.empty_cache()
#                    region_features_batch=torch.stack(region_features_batch,dim=0)
#                    region_features_batch=region_features_batch.to(device)

#                 # return torch.empty((0, 768), dtype=torch.float32, device=upscaled_features.device)   
#             # 将所有batch的特征堆叠成一个张量返回
#                region_features_batch=global_avg_pool(region_features_batch)
               
#                region_features_batch=mlp(region_features_batch.squeeze())
#                # 检查 region_features_batch 维度
#                if region_features_batch.dim() == 1:
#                    region_features_batch= region_features_batch.unsqueeze(0)
            
#                return region_features_batch






# # # Define the cross-modal mutual information loss function
# # # def cross_modal_mi_loss(v, l, tau):
# # #     """
# # #     v: visual feature of backbone
# # #     l: text description feature of the corresponding object
# # #     M: number of text description features
# # #     s: similarity score function between visual and text features
# # #     tau: temperature parameter
# # #     """
# # #     # logits = similarity_score(v, l) / tau  # compute similarity scores
# # #     # print(logits)
# # #     # print(logits.shape)
# # #     sim_matrix = torch.cosine_similarity(v.unsqueeze(1), l.unsqueeze(0), dim=-1)
# # #     # print("相似性矩阵的维度:",sim_matrix.shape)
# # #     # print(sim_matrix)
# # #     numerator = torch.exp(torch.diag(sim_matrix) / tau)
# # #     print("分子",numerator)
# # #     # print("分子",numerator)
# # #     #相似矩阵中，dim=0表示按照列求和，v,l, 相当于是语言向视觉靠拢
# # #     denominator1 = torch.sum(torch.exp(sim_matrix /tau), dim=0, keepdim=True)
# # #     print(denominator1)
# # #     #相似矩阵中，dim=1表示按照行求和，v,l, 相当于是视觉向语言靠拢
# # #     denominator2 = torch.sum(torch.exp(sim_matrix /tau), dim=1, keepdim=True)
# # #     # print("转换前：",denominator2)
# # #     denominator2=torch.transpose(denominator2, 0, 1)
# # #     print("denominator2",denominator2)
# # #     # print("denominator1",denominator1)
# # #     # print("转换后 denominator2",denominator2)
# # #     # print(numerator)
# # #     # print("分母",denominator1,denominator2.t())
# # #     # 计算 softmax 函数
# # #     softmax1 = numerator / denominator1
# # #     softmax2= numerator / denominator2
# # #     # print(softmax1,softmax2)
# # #      # 计算互信息
# # #     loss1 = torch.mean(-torch.log(softmax1))
# # #     loss2 =torch.mean(-torch.log(softmax2))
# # #     # print(loss1,loss2)
# # #     # print("损失",loss1,loss2)
# # #     # logits = logits.view(1, -1)  # flatten into a row vector
# # #     # # subtract the maximum value for numerical stability
# # #     # logits -= torch.max(logits, dim=-1, keepdim=True).values  
# # #     # probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdim=True) # compute probabilities
# # #     # loss = -torch.log(probs[:, 0]) # compute loss
# # #     return loss1,loss2

# # # Define the similarity score function
# # # def similarity_score(v, l):
# # #     """
# # #     Compute the similarity score between a visual feature v and a text description feature o using cosine similarity
# # #     v: visual feature of a whole image
# # #     l: text description feature of the corresponding object
# # #     """
# # #     return torch.nn.functional.cosine_similarity(v.unsqueeze(0), l.unsqueeze(0), dim=1)


# # # Compute the cross-modal mutual information loss
# # # loss = 0
# # # tau = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
# # # M =1 # only one text description
# # # s = similarity_score  # similarity
# # # Compute the loss
# # # loss = cross_modal_mi_loss(visual_feature, text_descriptions_features, s, tau)
# # # print(loss.item())

# @MODELS.register_module()
# class FasterRCNN(TwoStageDetector):
#     """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

#     def __init__(self,
#                  backbone: ConfigType,
#                  rpn_head: ConfigType,
#                  roi_head: ConfigType,
#                  train_cfg: ConfigType,
#                  test_cfg: ConfigType,
#                  neck: OptConfigType = None,
#                  data_preprocessor: OptConfigType = None,
#                  init_cfg: OptMultiConfig = None) -> None:
#         super().__init__(
#             backbone=backbone,
#             neck=neck,
#             rpn_head=rpn_head,
#             roi_head=roi_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             init_cfg=init_cfg,
#             data_preprocessor=data_preprocessor)
        
#         # self.DSR = FR()
        
#     def _forward(self, batch_inputs: Tensor,
#                  batch_data_samples: SampleList) -> tuple:
#         """Network forward process. Usually includes backbone, neck and head
#         forward without any post-processing.

#         Args:
#             batch_inputs (Tensor): Inputs with shape (N, C, H, W).
#             batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
#                 the meta information of each image and corresponding
#                 annotations.

#         Returns:
#             tuple: A tuple of features from ``rpn_head`` and ``roi_head``
#             forward.
#         """
#         results = ()
#         x = self.extract_feat(batch_inputs)
        

#         if self.with_rpn:
#             rpn_results_list = self.rpn_head.predict(
#                 x, batch_data_samples, rescale=False)
#         else:
#             assert batch_data_samples[0].get('proposals', None) is not None
#             rpn_results_list = [
#                 data_sample.proposals for data_sample in batch_data_samples
#             ]
#         roi_outs = self.roi_head.forward(x, rpn_results_list,
#                                          batch_data_samples)
#         results = results + (roi_outs, )
#         # print(results.shape)
#         # print(batch_data_samples)
#         return results

#     def loss(self, batch_inputs: Tensor,
#              batch_data_samples: SampleList) -> dict:
#         """Calculate losses from a batch of inputs and data samples.

#         Args:
#             batch_inputs (Tensor): Input images of shape (N, C, H, W).
#                 These should usually be mean centered and std scaled.
#             batch_data_samples (List[:obj:`DetDataSample`]): The batch
#                 data samples. It usually includes information such
#                 as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

#         Returns:
#             dict: A dictionary of loss components
#         """
 
#         x = self.extract_feat(batch_inputs)
        
        
#         # with open('/userHome/yym/code/mmdetection-3.x/out_put1.txt', 'w') as file:
#         #      file.write(np.array2string(x[0]))
#         # print("banckbone+fpn的特征",x1)
#         # print(x.shape)
#         x1 = self.backbone(batch_inputs)
#         # print("backbone的特征",x)
    
#         losses = dict()

#         x0 = x[0]
#         # print(x0.shape)
#         # x0=x0[:,:100,:,:]
#         # print(x0.shape)
#         # print(x0)
#         # x=x[0]
#         # print(x0.shape)
#         # print(x.shape)
#         # print("bankbone特征维度",x0.shape)
#             # 定义一个卷积***网络
#         # conv_net = nn.Sequential(
#         #    nn.Conv2d(256, 64, kernel_size=(3,3), stride=1, padding=1),
#         #    nn.ReLU(),
#         #    nn.MaxPool2d(kernel_size=(2,2)),
#         #    nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1),
#         #    nn.ReLU(),
#         #    nn.MaxPool2d(kernel_size=(2,2)),
#         #    nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=1),
#         #    nn.ReLU(),
#         #    nn.MaxPool2d(kernel_size=(2,2))
#         # )
#         # # 定义一个全连接网络
#         # fc_net = nn.Linear(256*25*42,768)
#         #两层的MLP
#         # class TwoLayerMLP(nn.Module):
#         #     def __init__(self, input_size, hidden_size, output_size):
#         #         super(TwoLayerMLP, self).__init__()
#         #         self.fc1 = nn.Linear(input_size, hidden_size)
#         #         self.fc2 = nn.Linear(hidden_size, output_size)
#         #         self.relu = nn.ReLU()

#         #     def forward(self, x):
#         #         # x = x.view(x.size(0), -1)  # Flatten the input to (batch_size, 256*25*42)
#         #         x = self.fc1(x)
#         #         x = self.relu(x)
#         #         x = self.fc2(x)
#         #         return x
#         # class TwoLayerMLP(nn.Module):
#         #     def __init__(self, hidden_size, output_size):
#         #        super(TwoLayerMLP, self).__init__()
#         #        self.fc1 = nn.Linear(hidden_size, hidden_size)
#         #        self.fc2 = nn.Linear(hidden_size, output_size)
#         #        self.relu = nn.ReLU()

#         #     def forward(self, x):
#         #         x = self.fc1(x)
#         #         x = self.relu(x)
#         #         x = self.fc2(x)
#         #         return x
#         # input_size=256*32*24
        
        

       

#         # output_feature = neck.forward_test(x0)
#         # print(output_feature.shape)
#         # visual_fature=mlp(x0)
#         # print(visual_fature.shape)


#         # # 进行前向传递
#         # conv_net=conv_net.to('cuda')
#         # x0 = conv_net(x0)
#         # # self.DSR.cuda()
#         # # x0 = self.DSR(x0)
#         # x0=x0.view(x0.size(0),-1)
#         # print(x0.shape)
#         # fc_net.cuda()
#         # visual_fature=fc_net(x0)
#         # print(visual_fature.shape)
        
#         # print("视觉特征的维度：",visual_fature.shape)
#         # print("视觉特征:",visual_fature)

#         #将batch里面的prompt全部取出
#         def extract_values(dictionary_list, key):
#             values = []
#             for dictionary in dictionary_list:
#                 if key in dictionary:
#                     values.append(dictionary[key])
#             return values
#         #判断batch_img_metas里面包含几张图像信息
#         # def count_nested_dicts(lst):
#         #     count = 0
#         #     for item in lst:
#         #         if isinstance(item, dict):
#         #             count += 1
#         #     return count
#         outputs=unpack_gt_instances(batch_data_samples)
#         (batch_gt_instances,batch_gt_instances_ignore,batch_img_metas)=outputs
#         # print("前面的数据：",batch_data_samples)
#         prompt=extract_values(batch_img_metas,'prompt')
#         # print("密集区域的prompt",prompt)
#         img_path=extract_values(batch_img_metas,'img_path')
#         flip_directions=extract_values(batch_img_metas,'flip_direction')
#         # print(img_path)
#         # print(prompt)
#         # text_descriptions_features=get_text_feature(prompt)
#         # print("文本特征：",text_descriptions_features)
#         # print("文本特征的维度:",text_descriptions_features.shape)
#         text_descriptions_features=get_text_feature_CLIP(prompt,1)
#         text_descriptions_features = [tensor.to(device) for tensor in text_descriptions_features]
#         text_descriptions_features=torch.cat(text_descriptions_features,dim=0)
#         sizes_img=extract_values(batch_img_metas,'ori_shape')
#         tau = 0.07
#         def cross_entropy(preds, targets, reduction='none'):
#                log_softmax = nn.LogSoftmax(dim=-1)
#                loss = (-targets * log_softmax(preds)).sum(1)
#                if reduction == "none":
#                    return loss
#                elif reduction == "mean":
#                    return loss.mean()
#         if len(text_descriptions_features)!=0:

           
#         #    print("批尺寸",sizes_img)
#         #    outsize=(size[0][1], size[0][0])
#         #    #将特征映射为原图尺寸
#         #    upscaled_features = F.interpolate(x0, size=outsize, mode='bilinear', align_corners=True)
#         #    print(x0.shape)
#         #    print(upscaled_features.shape)
#            dense_region=extract_values(batch_img_metas,'dense_region_bboxes')
#         #    print("密集区域",dense_region)
           

#            visual_feature=extract_dense_region_features(x0, dense_region,img_path,sizes_img,flip_directions)
          
           
#             # temperature hyper-parameter
#            text_descriptions_features=text_descriptions_features.half()
#            visual_feature=visual_feature.half()
#            logits = (text_descriptions_features.to(device) @ visual_feature.T.to(device)) / tau
#            images_similarity = visual_feature.to(device) @ text_descriptions_features.T.to(device)
#            texts_similarity = text_descriptions_features.to(device) @ visual_feature.T.to(device)
#         # logits = (text_descriptions_features@ visual_fature.T) / tau
#         # images_similarity = visual_fature @ text_descriptions_features.T
#         # texts_similarity = text_descriptions_features @ visual_fature.  #    
#            targets = F.softmax(
#            (images_similarity + texts_similarity) / 2 * tau, dim=-1
#         )
#            texts_loss = cross_entropy(logits, targets, reduction='none')
#            images_loss = cross_entropy(logits.T, targets.T, reduction='none')
#             #print("=======>",texts_loss,"fffff",images_loss)
#            loss_prompt =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
#         #    loss_prompt=images_loss
#            a=loss_prompt.tolist()
#         #    a=texts_loss.tolist()
#         #    b=images_loss.tolist()

#         # print(a)
#         # print("损失",loss_prompt)
#            if a != [] and a != [0.0]:
#                losses['loss_prompt']= (images_loss.mean()) * 0.005
#             #    print("不为0或者是空的损失",losses['loss_prompt_image'])
#         # loss_prompt1,loss_prompt2 = cross_modal_mi_loss(visual_fature.squeeze(0).to(device), text_descriptions_features.to(device), tau)
#         # # print("loss1:",loss1)
#         # losses['loss_prompt1'] = loss_prompt1
#         # losses['loss_prompt2'] = loss_prompt2
#         # else:
#         #     print("Prompt is empty. Skipping feature extraction and loss calculation.")
        


#         # if self.with_neck:
#         #     x = self.neck(x)
   
        

#         # RPN forward and loss
#         if self.with_rpn:
#             proposal_cfg = self.train_cfg.get('rpn_proposal',
#                                               self.test_cfg.rpn)
#             rpn_data_samples = copy.deepcopy(batch_data_samples)
#             # print(batch_data_samples)
#             # set cat_id of gt_labels to 0 in RPN
#             for data_sample in rpn_data_samples:
#                 data_sample.gt_instances.labels = \
#                     torch.zeros_like(data_sample.gt_instances.labels)

#             rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
#                 x, rpn_data_samples, proposal_cfg=proposal_cfg)
#             # avoid get same name with roi_head loss
#             # print(rpn_results_list)
#             keys = rpn_losses.keys()
#             for key in list(keys):
#                 if 'loss' in key and 'rpn' not in key:
#                     rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
#             losses.update(rpn_losses)
#         else:
#             assert batch_data_samples[0].get('proposals', None) is not None
#             # use pre-defined proposals in InstanceData for the second stage
#             # to extract ROI features.
#             rpn_results_list = [
#                 data_sample.proposals for data_sample in batch_data_samples
#             ]

#         roi_losses = self.roi_head.loss(x, rpn_results_list,
#                                         batch_data_samples)
#         # roi_outs=self.roi_head.forward(x,rpn_results_list,batch_data_samples)
#         # print(roi_outs)
#         # print(rpn_results_list)
#         # print(batch_data_samples)
        
#         # difficult_object =extract_values(batch_img_metas,'difficult_object_bboxes')
#         # object_prompt=extract_values(batch_img_metas,'object_prompt')
#         # # print("困难目标的prompt",object_prompt)
#         # if difficult_object and any(difficult_object):
#         #     # print(difficult_object)
#         #     object_prompt_feature=get_text_feature_CLIP(object_prompt,0)
#         #     object_visual_feature=extract_dense_region_features(x0, difficult_object,img_path,sizes_img)
#         #     logits = (object_prompt_feature.to(device) @ object_visual_feature.T.to(device)) / tau
#         #     images_similarity_object = object_visual_feature.to(device) @ object_prompt_feature.T.to(device)
#         #     texts_similarity_object = object_prompt_feature.to(device) @ object_visual_feature.T.to(device)
#         # # logits = (text_descriptions_features@ visual_fature.T) / tau
#         # # images_similarity = visual_fature @ text_descriptions_features.T
#         # # texts_similarity = text_descriptions_features @ visual_fature.  #    
#         #     targets = F.softmax(
#         #    (images_similarity_object + texts_similarity_object) / 2 * tau, dim=-1
#         # )
#         #     # texts_loss = cross_entropy(logits, targets, reduction='none')
#         #     loss_object = cross_entropy(logits.T, targets.T, reduction='none')
#         #     b=loss_object.tolist()
#         #     if b != [] and b != [0.0]:
#         #         losses['loss_prompt_object']= (loss_object.mean()) * 0.008
#                 # print(losses['loss_prompt_object'])
                
#             #print("=======>",texts_loss,"fffff",images_loss)
#             # loss_prompt =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
#         # print(object_prompt_feature.shape)
#         # print(object_visual_feature.shape)
#         # print("困难目标：",diffcult_object)
#         # print("困难目标的prompt: ",object_prompt)
#         # print(rpn_results_list)
        
#         losses.update(roi_losses)
#         # print(losses)
#         return losses


