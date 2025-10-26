from PIL import Image, ImageOps
from mmdet.utils.setup_env import register_all_modules
from mmdet.apis import inference_detector


def anomaly_region_location(detector_model, frame_path):
    register_all_modules(True)
    result = inference_detector(model=detector_model, imgs=frame_path);
    merged_bbox = None

    # 遍历所有检测到的方框
    if result.pred_instances.bboxes is not None:
        for bbox in result.pred_instances.bboxes:
            x1, y1, x2, y2 = bbox.tolist()  # 获取方框的左上角和右下角坐标，以及置信度
            if merged_bbox is None:
                # 第一个方框，直接赋值给merged_bbox
                merged_bbox = [x1, y1, x2, y2]
            else:
                # 后续方框，更新merged_bbox以包含当前方框
                merged_bbox[0] = min(merged_bbox[0], x1)  # 更新左上角x坐标
                merged_bbox[1] = min(merged_bbox[1], y1)  # 更新左上角y坐标
                merged_bbox[2] = max(merged_bbox[2], x2)  # 更新右下角x坐标
                merged_bbox[3] = max(merged_bbox[3], y2)  # 更新右下角y坐标
    if merged_bbox is None:
        return 0
    else:
        return 1


def check_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            if len(data.strip()) == 0:  # 判断文件内容是否为空
                return 0
            else:
                return 1
    except FileNotFoundError:
        print("文件不存在")
        return -1
