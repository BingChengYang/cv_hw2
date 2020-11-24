import random
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
fruits_nuts_metadata=MetadataCatalog.get('dataset_train')
print(fruits_nuts_metadata)


# img = cv2.imread('./dataset_coco/images/1.pngs')
# visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
# vis = visualizer.draw_dataset_dict(d)
# cv2_imshow(vis.get_image[:, :, ::-1])

