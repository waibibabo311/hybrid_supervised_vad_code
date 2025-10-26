from pre_feature_extracter.utils import anomaly_region_location, check_txt_file

frame_path="/data/gzh/Ubnormal_For_OD/Scene23/abnormal_scene_23_scenario_3_frames/abnormal_scene_23_scenario_3_0431_frame.png";
gt_path = frame_path.replace("_frames", "_annotations").replace("_frame.png", "_gt.txt")

print(check_txt_file(gt_path))