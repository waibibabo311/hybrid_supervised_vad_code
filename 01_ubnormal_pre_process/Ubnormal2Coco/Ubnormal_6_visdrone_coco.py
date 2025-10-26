import os
import cv2
from tqdm import tqdm
import json


def test():
    dir=r'D:\pythonProjects\Test\visdrone2coco'
    train_dir = os.path.join(dir, "annotations")
    print(train_dir)
    id_num = 0
    categories = [

        {"id": 0, "name": "AnomalyRegion"}
    ]
    images = []
    annotations = []
    set = os.listdir('/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Coco_20240229/train/annotations')
    annotations_path = '/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Coco_20240229/train/annotations'
    images_path = '/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Coco_20240229/train/images'

    for i in tqdm(set):
        print(annotations_path + "/" + i, "r")
        f = open(annotations_path + "/" + i, "r")
        name = i.replace(".txt", "")
        image = {}
        height, width = cv2.imread(images_path + "/" + name + ".jpg").shape[:2]
        file_name = name + ".jpg"
        image["file_name"] = file_name
        image["height"] = height
        image["width"] = width
        image["id"] = name
        images.append(image)
        for line in f.readlines():
            annotation = {}
            line = line.replace("\n", "")
            if line.endswith(","):  # filter data
                line = line.rstrip(",")
            line_list = [int(i) for i in line.split(",")]
            bbox_xywh = [line_list[0], line_list[1], line_list[2], line_list[3]]
            annotation["image_id"] = name
            annotation["score"] = line_list[4]
            annotation["bbox"] = bbox_xywh
            annotation["category_id"] = int(line_list[5])-1
            annotation["id"] = id_num
            annotation["iscrowd"] = 0
            annotation["segmentation"] = []
            annotation["area"] = bbox_xywh[2] * bbox_xywh[3]
            id_num += 1
            print(id_num)
            annotations.append(annotation)
        dataset_dict = {}
        dataset_dict["images"] = images
        dataset_dict["annotations"] = annotations
        dataset_dict["categories"] = categories
        json_str = json.dumps(dataset_dict)
        with open(f'/data/gzh/UBnormal_Origin_For_OD_20240227/UBnormal_Coco_20240229/train/output/output.json', 'w') as json_file:
            json_file.write(json_str)
    print("json file write done...")

if __name__ == '__main__':
    test()
