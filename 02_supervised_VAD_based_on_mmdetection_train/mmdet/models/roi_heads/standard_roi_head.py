# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import cv2
import os
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import logging
@MODELS.register_module()

class StandardRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = MODELS.build(mask_head)



    # TODO: Need to refactor later
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        # print(rpn_results_list)
        # print(proposals)
        # print(batch_data_samples)
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs
        num_imgs = len(batch_data_samples)
        sampling_results = []
        pos_proposals=[]
        neg_proposals=[]
        def extract_values(dictionary_list, key):
            values = []
            for dictionary in dictionary_list:
                if key in dictionary:
                    values.append(dictionary[key])
            return values
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            # print(sampling_result)
            pos_proposal=sampling_result.pos_bboxes
            neg_proposal=sampling_result.neg_bboxes
            # neg_proposal=rpn_results.priors

           
            # print("rpn_results_____________",len(rpn_results.priors))
            # img_path=extract_values(batch_img_metas,'img_path')
            # print(img_path)
            # print("正样本：",pos_proposal)
            pos_proposals.append(pos_proposal)
            neg_proposals.append(neg_proposal)
            sampling_results.append(sampling_result)
        
        
        # difficult_objects =extract_values(batch_img_metas,'difficult_object_bboxes')
        # easy_object=extract_values(batch_img_metas,'easy_object')
        # img_path=extract_values(batch_img_metas,'img_path')
        # scale_factors=extract_values(batch_img_metas,'scale_factor')
        # flip_directions=extract_values(batch_img_metas,'flip_direction')
        # pad_shapes=extract_values(batch_img_metas,'pad_shape')
        # object_prompt=extract_values(batch_img_metas,'object_prompt')
        # prompt=extract_values(batch_img_metas,'prompt')
        # dense_region=extract_values(batch_img_metas,'dense_region_bboxes')
        # # print("密集区域:",dense_region)
        # # print("困难目标:",difficult_objects)
        # # Merge the dense regions into a single list
        # # print(merged_dense_regions)
        # # print(img_path)
        # # print("目标prompt",object_prompt)
        # # print(batch_img_metas)
        # # print("易检测目标:",easy_object)
        # labels=extract_values(batch_gt_instances,"labels")
        # bboxes_gts=extract_values(batch_gt_instances,"bboxes")
        # # print(label,bboxes_gt)

        # # print("困难目标：",difficult_objects)
        # # object_prompt=extract_values(batch_img_metas,'object_prompt')
        # import os
        # def get_text_feature_CLIP(prompts):
        #     # 定义本地目录来存储预训练模型权重
        #     # model_dir = "/userHome/yym/code/mmdetection-3.x/CLIP"
        #     # logging.set_verbosity_error()#去除警告

        #     # 加载本地预训练模型权重
           
        #     # processor = CLIPProcessor.from_pretrained(model_dir)
        #     # model = CLIPModel.from_pretrained(model_dir)
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
        #             # print(text)
        #             # inputs = processor(text, return_tensors="pt", padding=True, truncation=True)
        #             # text_feature = model.get_text_features(**inputs)
        #             text_input=clip.tokenize(text).to(device)
        #             text_feature = model.encode_text(text_input)
        #             # print(text_feature.shape)
        #             text_features.append(text_feature)     
        #     return  text_features   
        # # print(len(text_dense_feature))
        # # print(text_dense_feature.shape)


        # def get_clip_features(visual_inputs, text_inputs):
        #     # Define CLIP processing and model
        #     model, preprocess = clip.load("ViT-B/32", device=device)

        #     visual_features_list = []
        #     text_features_list = []

        #     # Process each visual and text input and get features
        #     for visual_input, text_input in zip(visual_inputs, text_inputs):
        #         # Visual Input Processing
        #         # print(visual_input)
        #         # print(text_input)
        #         if not text_input:
        #             text_features_list.append(torch.tensor([]))
        #         else:
        #             text_input=clip.tokenize(text_input).to(device)
        #             text_features = model.encode_text(text_input)
        #         # print(text_features.shape)
                
        #         text_features_list.append(text_features)
        #         if not visual_input:
        #             visual_features_list.append(torch.tensor([]))
        #             continue
        #         image = preprocess(Image.open(visual_input)).unsqueeze(0).to(device)
        #         image_features = model.encode_image(image)
        #         # print(image_features)
        #         visual_features_list.append(image_features)
        #         # print(text_input)
        #     return visual_features_list , text_features_list

        # def batch_extract_and_get_clip_features(image_paths, target_bboxes_list, texts_list, save_dir="cropped_images"):
        #     # Create a directory for saving cropped images
        #     os.makedirs(save_dir, exist_ok=True)

        #     # Define CLIP processing and model
        #     model_path = "/userHome/yym/code/mmdetection-3.x/CLIP"
        #     processor = CLIPProcessor.from_pretrained(model_path)
        #     model = CLIPModel.from_pretrained(model_path)
         
        #     visual_inputs_list = []  # List to store visual inputs for each image
        #     text_inputs_list = texts_list  # Assume texts are provided for each bounding box in the batch

        #     # Process each image and its corresponding bounding boxes
        #     for image_path, target_bboxes in zip(image_paths, target_bboxes_list):
        #         visual_inputs = []  # List to store visual inputs for each image

        #         # Open the image
        #         image = Image.open(image_path)

        #         # Create a subdirectory for each image
        #         image_name = os.path.splitext(os.path.basename(image_path))[0]
        #         image_save_dir = os.path.join(save_dir, image_name)
        #         os.makedirs(image_save_dir, exist_ok=True)

        #         # Process each bounding box for the current image

        #         for i, bbox in enumerate(target_bboxes):
        #             if not bbox:
        #                 visual_inputs.append([])
        #                 continue

        #             # Crop the image based on the bounding box
        #             cropped_image = image.crop((bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]))

        #             # Save the cropped image to the local filesystem
        #             save_path = os.path.join(image_save_dir, f"cropped_image_{i}.png")
        #             cropped_image.save(save_path)

        #             # Append the cropped image path to the visual inputs for the current image
        #             visual_inputs.append(save_path)

        #         # Append the visual inputs for the current image to the overall list
        #         visual_inputs_list.append(visual_inputs)
        #         # print(visual_inputs_list)

        #     # Process visual and text inputs and get features
        #     visual_features_list, text_features_list = [], []

        #     for visual_inputs,text_inputs in zip(visual_inputs_list,text_inputs_list):
        #         # print(text_inputs)
        #         if text_inputs:
        #             visual_features, text_features = get_clip_features(visual_inputs,text_inputs )
        #             visual_features_list.append(visual_features)
        #             text_features_list.append(text_features)
        #         else:
        #             visual_features_list.append([])
        #             text_features_list.append([])


        #     # Stack visual and text features for the entire batch
        #     # visual_features_tensor = torch.cat(visual_features_list, dim=0)
        #     # text_features_tensor = torch.cat(text_features_list, dim=0)

        #     return visual_features_list, text_features_list
        
        losses = dict()
        # t = torch.Tensor([1, 2])

        # losses['roi_prompt']=t
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            # print(bbox_results['bbox_feats'].shape)
            losses.update(bbox_results['loss_bbox'])
        
        # print("bbox_results",bbox_results['bbox_feats'].shape)
        # print(rpn_results)
        # 328-482
        # output_dir='/userHome/yym/code/mmdetection-3.x/visual_proposal'
        # proposals = [rpn_results.priors for rpn_results in rpn_results_list]
        # # print(neg_proposals)
        # def  resize_bbox(difficult_objects,scale_factors,img_paths,flip_directions):
        #     difflcult_object_resize_lists=[]
        #     for difflcult_object,scale_factor,img_path,flip_direction in zip(difficult_objects,scale_factors,img_paths,flip_directions):
        #         image=cv2.imread(img_path)
        #         original_width=image.shape[1]
        #         mid=original_width//2
        #         difflcult_object_resize_list=[]
        #         for  proposal in difflcult_object:
        #             if len(proposal)==4:
        #                 x1, y1, w, h = proposal
        #             else:
        #                 x1, y1, w, h,c = proposal  # 提议框的坐标信息
        #             x2=x1+w
        #             y2=y1+h
        #             # print("原始的尺寸：",x1, y1, x2, y2)                    
        #             scale_x=scale_factor[1]
        #             scale_y=scale_factor[0]
                    
        #             x1_original = x1 * scale_x
        #             y1_original = y1 * scale_y
        #             x2_original = x2 * scale_x
        #             y2_original = y2 * scale_y
                    
        #             if flip_direction=="horizontal":
        #                 x1_original=2*mid-x1_original
        #                 x2_original=2*mid-x2_original
        #             difflcult_object_resize=[x1_original,y1_original,x2_original,y2_original]
        #             difflcult_object_resize_list.append(difflcult_object_resize)

        #             # print(x1_original,y1_original,x2_original,y2_original)
        #         difflcult_object_resize_lists.append(difflcult_object_resize_list)
        #     return difflcult_object_resize_lists
        
        # class GlobalAvgPool2d(nn.Module):
        #     def forward(self, x):
        # # 进行全局平均池化操作
        #         return nn.functional.adaptive_avg_pool2d(x.unsqueeze(0), (1, 1)).squeeze(-1).squeeze(-1)
        # def cross_entropy(preds, targets, reduction='none'):
        #         log_softmax = nn.LogSoftmax(dim=-1)
        #         loss = (-targets * log_softmax(preds)).sum(1)
        #         if reduction == "none":
        #             return loss
        #         elif reduction == "mean":
        #             return loss.mean()
        # output_size = 512
        # global_avg_pool = GlobalAvgPool2d()
        # global_avg_pool.to(device)
        # mlp = nn.Linear(256, output_size)
        # mlp.to(device)
        # dense_region=resize_bbox(dense_region,scale_factors,img_path,flip_directions) 
        # dense_region=[torch.tensor(image_data) for image_data in dense_region]
        # print(dense_region)
        # dense_region = bbox2roi(dense_region)
        # bbox_forward_results=self._bbox_forward(x,dense_region)
        # dense_region_feature=bbox_forward_results['bbox_feats']
        # dense_region_feature=global_avg_pool(dense_region_feature)
        # # print("batch视觉特征维度",visual_features_batch.shape)
        # if dense_region_feature.shape == torch.Size([1, 1, 256]):
        #     dense_region_feature= dense_region_feature.squeeze().unsqueeze(0)
        #     # print("转换后前的维度：",visual_features_batch.shape)
        #     dense_region_feature=mlp(dense_region_feature)
        #     # print("转换后的维度：",visual_feature.shape)
        # else:
        #     dense_region_feature=mlp(dense_region_feature.squeeze())
        # dense_text_feature=get_text_feature_CLIP(prompt)
        # tau = 0.07
        # dense_text_feature= [tensor.to(device) for tensor in dense_text_feature]
        # dense_text_feature=torch.cat(dense_text_feature,dim=0)

        # # temperature hyper-parameter
        # dense_text_feature=dense_text_feature.half()
        # # print(text_dense_feature.shape)
        # dense_region_feature=dense_region_feature.half()
        # logits = (dense_text_feature.to(device) @ dense_region_feature.T.to(device)) / tau

        # images_similarity = dense_region_feature.to(device) @ dense_text_feature.T.to(device)
        # texts_similarity = dense_text_feature.to(device) @ dense_region_feature.T.to(device)
        # # logits = (text_descriptions_features@ visual_fature.T) / tau
        # # images_similarity = visual_fature @ text_descriptions_features.T
        # # texts_similarity = text_descriptions_features @ visual_fature.  #    
        # targets = F.softmax(
        # (images_similarity + texts_similarity) / 2 * tau, dim=-1
        # )
        # texts_loss = cross_entropy(logits, targets, reduction='none')
        # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        #     #print("=======>",texts_loss,"fffff",images_loss)
        # loss_prompt =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        # #    loss_prompt=images_loss
        # a=loss_prompt.tolist()
        # # print(a)
        # if a != [] and a != [0.0]:
        #     losses['loss_prompt']= (images_loss.mean()) * 0.001

        # if any(sublist for sublist in difficult_objects):
        #     # visual_features_list, text_features_list=batch_extract_and_get_clip_features(img_path,easy_object,object_prompt)     
        #     # # 将二维列表展平为一维列表
        #     # # 使用列表推导式将二维列表展平并确保在相同的设备上
        #     # flat_visual_features = [item.flatten().to(device) for sublist in visual_features_list for item in sublist]
        #     # flat_text_features = [item.flatten().to(device) for sublist in text_features_list for item in sublist]
        #     # text_features=torch.stack(flat_text_features,dim=0)
        #     # concatenated_features_list = [torch.cat([flat_visual, flat_text], dim=0) for flat_visual, flat_text in zip(flat_visual_features, flat_text_features)]
        #     text_dense_feature=get_text_feature_CLIP(object_prompt)
        #     # print(len(text_dense_feature))
        #     text_dense_feature = [tensor.to(device) for tensor in text_dense_feature]
        #     text_dense_feature=torch.cat(text_dense_feature,dim=0)
        #     difflcult_object_resize_lists=resize_bbox(difficult_objects,scale_factors,img_path,flip_directions) 
            
        #     tensor_difficult = [torch.tensor(image_data) for image_data in difflcult_object_resize_lists]
        
        #     # print("困难目标",difficult_objects)
        #     # print(len(difficult_objects))
        #     rois = bbox2roi(tensor_difficult)
        #     # print(rois)
            
            

        #     bbox_forward_results=self._bbox_forward(x,rois)
        #     difficult_objects_visual_feature=bbox_forward_results['bbox_feats']
        #     visual_features_batch=global_avg_pool(difficult_objects_visual_feature)
        #     # print("batch视觉特征维度",visual_features_batch.shape)
        #     if visual_features_batch.shape == torch.Size([1, 1, 256]):
        #         visual_features_batch= visual_features_batch.squeeze().unsqueeze(0)
        #         # print("转换后前的维度：",visual_features_batch.shape)
        #         visual_feature=mlp(visual_features_batch)
        #         # print("转换后的维度：",visual_feature.shape)
        #     else:
        #         visual_feature=mlp(visual_features_batch.squeeze())
        #     # print("视觉特征的维度：",visual_feature.shape
        #      # temperature hyper-parameter
        #     text_object_feature=text_dense_feature.half()
        #     # print(text_dense_feature.shape)
        #     visual_feature=visual_feature.half()
        #     logits = (text_object_feature.to(device) @ visual_feature.T.to(device)) / tau

        #     images_similarity = visual_feature.to(device) @ text_object_feature.T.to(device)
        #     texts_similarity = text_object_feature.to(device) @ visual_feature.T.to(device)
        #     # logits = (text_descriptions_features@ visual_fature.T) / tau
        #     # images_similarity = visual_fature @ text_descriptions_features.T
        #     # texts_similarity = text_descriptions_features @ visual_fature.  #    
        #     targets = F.softmax(
        #     (images_similarity + texts_similarity) / 2 * tau, dim=-1
        #     )
        #     texts_loss = cross_entropy(logits, targets, reduction='none')
        #     images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        #         #print("=======>",texts_loss,"fffff",images_loss)
        #     loss_prompt =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        #     #    loss_prompt=images_loss
        #     a=loss_prompt.tolist()
        #     # print(a)
        #     if a != [] and a != [0.0]:
        #        losses['loss_object']= (images_loss.mean()) * 0.005
            # print(bbox_forward_results['bbox_feats'].shape)
        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])
        # print(losses)
        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        # print(rois)
        bbox_results = self._bbox_forward(x, rois)
        # print("bbox_results",bbox_results)
        # print(sampling_results)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)
        return mask_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        # print(result_list)
        return result_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list
