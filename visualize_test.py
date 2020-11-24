from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import random
import cv2
import numpy as np
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from os import listdir
from os.path import isfile, join
import json
dir = os.listdir('./test')
dir.sort()
dir.sort(key = lambda x: int(x[:-4]))


register_coco_instances("dataset_train", {}, "./dataset_coco/annotations/train.json", "./dataset_coco/images")

imgs = [f for f in dir if isfile(join('./test/', f))]

cfg = get_cfg()
cfg.merge_from_file(
    "./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
)
cfg.DATASETS.TEST = ("dataset_train",)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3# Threshold
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0001
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.DEVICE = "cuda" # cpu or cuda
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # 3 classes (data, fig, hazelnut)

predictor = DefaultPredictor(cfg)
print(MetadataCatalog.get('dataset_train'))
# predictor.test()
ans = []
for i in range(10):
    print(i)
    dict_ans = {'bbox': [], 'score': [], 'label': []}
    image = cv2.imread(os.path.join("./test/", imgs[i]))
    # Make prediction
    output = predictor(image)
    bbox = (output['instances'].pred_boxes).tensor
    bbox = (bbox.cpu().numpy()).tolist()
    
    for k in range(len(bbox)):
        bbox[k][0], bbox[k][1] = bbox[k][1], bbox[k][0]
        bbox[k][2], bbox[k][3] = bbox[k][3], bbox[k][2]
    score = output['instances'].scores
    score = score.cpu().numpy()
    label = output['instances'].pred_classes
    label = (label.cpu().numpy()).tolist()
    for k in range(len(label)):
        if label[k] == 0:
            label[k] = 10
    v = Visualizer(image[:, :, ::-1],
                scale=1,
                metadata=MetadataCatalog.get('dataset_train'),
                instance_mode=ColorMode.IMAGE
                )
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    cv2.imshow('images', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    dict_ans['bbox'] = bbox
    dict_ans['score'] = score.tolist()
    dict_ans['label'] = label
    ans.append(dict_ans)
print("done")

# json_name = os.path.join('./submit/', '0756545.json')
# with open(json_name, 'w') as f:
#     json.dump(ans, f)