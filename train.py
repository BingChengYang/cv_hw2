from detectron2.data.datasets import register_coco_instances
import random
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
import argparse
register_coco_instances("dataset_train", {}, "./dataset_coco/annotations/train.json", "./dataset_coco/images")

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.00015)
parser.add_argument("--mini_batch_size", default=32)
parser.add_argument("--epoches", default=5000)
args = parser.parse_args()
cfg = get_cfg()
cfg.merge_from_file(
    "./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = args.lr
cfg.SOLVER.MAX_ITER = (
    args.epoches
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    args.mini_batch_size
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()