# cv_hw2

* 1.Prepare Dataset
* 2.Training
* 3.Testing

## Prepare Dataset
先使用下列指令將digitStruct.mat轉成train_data_processed.h5格式
```
python construct_datasets.py
```
接下來需要將train_data_processed.h5轉換成coco dataset的格式
```
python make_coco_dataset.py
```
會產生coco格式的train.json

## Training model
使用以下指令訓練模型:
```
python train.py --lr=0.00015 --epoches=500 --mini_batch_size=32
```
lr代表learning rate的大小，default = 0.00015</br>
epoches代表總共訓練幾個epoch，default = 500</br>
mini_batch_size代表會使用mini_batch的大小，default = 32</br>

## Testing model
使用下列指令來產生預測結果:
```
python test.py --test_model="model_final.pkl"
```
test_model代表所要選擇測試的model，default = "model_final.pkl"</br>
結束後會在submit資料夾產生0756545.json
