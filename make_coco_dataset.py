import os
import json
import pandas as pd
import cv2


dataset = {'images':[], 'categories':[], 'annotations':[]}

for i in range(10):
    if i == 0:
        dataset['categories'].append({'id': 0, 'name': str(10), 'supercategory': 'digit'})
    else:
        dataset['categories'].append({'id': i, 'name': str(i), 'supercategory': 'digit'})

test = pd.read_hdf('./dataset_coco/images/train_data_processed.h5','table')
print(test)

train_data = (pd.read_hdf('./dataset_coco/images/train_data_processed.h5','table')).values

for i in range(len(train_data)):
    img = cv2.imread(os.path.join('./dataset_coco/images', train_data[i][1]))
    dataset['images'].append({
        'file_name': train_data[i][1],
        'id': i,
        'width': img.shape[1],
        'height': img.shape[0]
    })


for i in range(len(train_data)):
    if int(train_data[i][2]) == 10:
        label = 0
    else:
        label = int(train_data[i][2])
    x_min = float(train_data[i][3])
    x_max = float(train_data[i][7])
    y_min = float(train_data[i][4])
    y_max = float(train_data[i][6])
    box_width =  max(x_max-x_min, 0)
    box_height = max(y_max-y_min, 0)
    dataset['annotations'].append({
        'area': box_height * box_width,
        'bbox': [x_min, y_min, box_width, box_height],
        'category_id': label,
        'id': i,
        'image_id' : i,
        'iscrowd': 0,
        'segmentation': [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]
    })

folder = os.path.join('./dataset_coco/', 'annotations')
json_name = os.path.join(folder, 'train.json')
with open(json_name, 'w') as f:
    json.dump(dataset, f)