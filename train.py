from detectron2.data.datasets import register_coco_instances
import random
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

register_coco_instances("dataset_train", {}, "./dataset_coco/annotations/train.json", "./dataset_coco/images")


cfg = get_cfg()
cfg.merge_from_file(
    "./configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml"
)
cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00015
cfg.SOLVER.MAX_ITER = (
    5000
)  # 300 iterations seems good enough, but you can certainly train longer
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
#     64
# )  # faster, and good enough for this toy dataset
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 # 3 classes (data, fig, hazelnut)
cfg.MODEL.RETINANET.BATCH_SIZE_PER_IMAGE = (
    32
)  # faster, and good enough for this toy dataset
cfg.MODEL.RETINANET.NUM_CLASSES = 10 # 3 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()