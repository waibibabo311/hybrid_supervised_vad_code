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
    # print(merged_bbox)
    region, background = crop_and_zero_out(Image.open(frame_path), merged_bbox)
    region = resize_with_padding(region, (256, 256))
    background = resize_with_padding(background, (256, 256))
    return region, background


def crop_and_zero_out(image, roi):

    if roi is None:
        return image, image

    # 提取 ROI 坐标
    left, upper, right, lower = roi

    # 裁剪图像
    roi_image = image.crop((int(left), int(upper), int(right), int(lower)))

    # 返回裁剪下来的 ROI 图像和零填充后的原图
    return roi_image, image


def resize_with_padding(img, target_size):
    # 计算目标尺寸
    width, height = img.size
    aspect_ratio = width / height
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height

    # 根据目标尺寸调整大小
    if aspect_ratio > target_aspect_ratio:
        # 基于高度进行调整
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        # 基于宽度进行调整
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

    # 计算填充的边缘
    padding_width = (target_width - new_width) // 2
    padding_height = (target_height - new_height) // 2

    # 创建填充后的图像
    padded_img = ImageOps.pad(resized_img, size=(target_width, target_height), color=0)

    return padded_img

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

