import torch
from mmdet.apis import init_detector, inference_detector
from pre_feature_extracter.utils import anomaly_region_location


region_location_config_file = "/userHome/gzh/mmdetection-3.x_20240307/NewFolder/faster-rcnn_r50_fpn_2x_coco.py"
region_location_checkpoint_file = '/userHome/gzh/mmdetection-3.x_20240307/work_dirs/prompt/epoch_313.pth'

region_location_model = init_detector(config=region_location_config_file, checkpoint=region_location_checkpoint_file,
                          device='cuda:0')

region, background = anomaly_region_location(region_location_model, rf'/data/gzh/Ubnormal_For_OD/Scene3/abnormal_scene_3_scenario_4_frames/abnormal_scene_3_scenario_4_0384_frame.png')
region.save("/userHome/gzh/temp/region.png")
background.save("/userHome/gzh/temp/background.png")