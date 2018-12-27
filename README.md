# YOLO3 (Detection, Training, and Evaluation)

## Dataset and Model

Dataset | mAP | Demo | Config | Model
:---:|:---:|:---:|:---:|:---:
Kangaroo Detection (1 class) (https://github.com/experiencor/kangaroo) | 95% | https://youtu.be/URO3UDHvoLY | check zoo | http://bit.do/ekQFj
Raccoon Detection (1 class) (https://github.com/experiencor/raccoon_dataset) | 98% | https://youtu.be/lxLyLIL7OsU | check zoo | http://bit.do/ekQFf
Red Blood Cell Detection (3 classes) (https://github.com/experiencor/BCCD_Dataset) | 84% | https://imgur.com/a/uJl2lRI | check zoo | http://bit.do/ekQFc
VOC (20 classes) (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) | 72% | https://youtu.be/0RmOI6hcfBI | check zoo | http://bit.do/ekQE5

## Todo list:
- [x] Yolo3 detection
- [x] Yolo3 training (warmup and multi-scale)
- [x] mAP Evaluation
- [x] Multi-GPU training
- [x] Evaluation on VOC
- [ ] Evaluation on COCO
- [ ] MobileNet, DenseNet, ResNet, and VGG backends

## Detection

Grab the pretrained weights of yolo3 from https://pjreddie.com/media/files/yolov3.weights.

```python yolo3_one_file_to_detect_them_all.py -w yolo3.weights -i dog.jpg``` 
***这个利用在COCO训练好的模型来做预测，但没有将怎么在COCO上面训练**

## Training

### 1. Data preparation 

Download the Raccoon dataset from from https://github.com/experiencor/raccoon_dataset.

Organize the dataset into 4 folders:

+ train_image_folder <= the folder that contains the train images.

+ train_annot_folder <= the folder that contains the train annotations in VOC format.

+ valid_image_folder <= the folder that contains the validation images.

+ valid_annot_folder <= the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.

### 2. Edit the configuration file
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "min_input_size":       352,
        "max_input_size":       448,
        "anchors":              [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326],
        "labels":               ["raccoon"]
    },

    "train": {
        "train_image_folder":   "/home/andy/data/raccoon_dataset/images/",
        "train_annot_folder":   "/home/andy/data/raccoon_dataset/anns/",      
          
        "train_times":          10,             # the number of time to cycle through the training set, useful for small datasets
        "pretrained_weights":   "",             # specify the path of the pretrained weights, but it's fine to start from scratch
        "batch_size":           16,             # the number of images to read in each batch
        "learning_rate":        1e-4,           # the base learning rate of the default Adam rate scheduler
        "nb_epoch":             50,             # number of epoches
        "warmup_epochs":        3,              # the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically
        "ignore_thresh":        0.5,
        "gpus":                 "0,1",

        "saved_weights_name":   "raccoon.h5",
        "debug":                true            # turn on/off the line that prints current confidence, position, size, class losses and recall
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",

        "valid_times":          1
    }
}

```

The ```labels``` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network. The rest images are simply ignored. By this way, a Dog Detector can easily be trained using VOC or COCO dataset by setting ```labels``` to ```['dog']```.

Download pretrained weights for backend at:

https://1drv.ms/u/s!ApLdDEW3ut5fgQXa7GzSlG-mdza6

**This weights must be put in the root folder of the repository. They are the pretrained weights for the backend only and will be loaded during model creation. The code does not work without this weights.**

### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

### 4. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 5. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

## Evaluation

`python evaluate.py -c config.json`

Compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.

## 自己的实验记录

1. Inference没有问题，用的虚拟环境是MRCNN
2. 训练Raccoon数据集的时候，总是在epoch=10左右的时候就early stop了，mAP=0.59左右，远低于给出的mAP。
   费了好大功夫，终于下载到完整的backend.h，终于可以正常的训练了。
   - 用backend训练了30 epochs，early stop,mAP 0.9048，还是很有差距。
   - 将batch size由4改成8，mAP为0.9737
   - lr有1e-4改成1e-3,mAP=0.9756
3. 训练VOC
   - 平台：2070，虚拟环境：MRCNN
   - lr 1e-4 mAP=0.6258
     lr 5e-4 mAP=0.5353
4. KITTI训练结果[将labels合并成三类]
    1. lr:1e-4 其他参数默认
    Epoch 00031: loss did not improve from 8.93820
    Epoch 00031: early stopping
    Car: 0.8372
    Cyclist: 0.5783
    Pedestrian: 0.5841
    mAP: 0.6665
    2. 重新生成anchors:7,66, 10,28, 15,116, 21,45, 31,192, 37,68, 61,107, 83,182, 123,237
    Epoch 00051: loss did not improve from 11.21383
    Epoch 00051: early stopping
    Car: 0.8486
    Cyclist: 0.6377
    Pedestrian: 0.5881
    mAP: 0.6915
 5. KITTI 忽略Mic类和Not Care类，其余7类
    1. 参数跟4.2相同
    Epoch 00025: loss did not improve from 13.12234
    Epoch 00025: early stopping
    Car: 0.8017
    Cyclist: 0.4698
    Pedestrian: 0.5721
    Person_sitting: 0.1993
    Tram: 0.6133
    Truck: 0.7080
    Van: 0.4755
    mAP: 0.5485
    2. 在5.1的基础上去掉Early Stop,再训练10epochs，其余参数跟5.1相同，表现有所提升，但是其实早就进入eraly stop了
    Epoch 00010: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-07.
    Car: 0.8119
    Cyclist: 0.5271
    Pedestrian: 0.5585
    Person_sitting: 0.2144
    Tram: 0.7243
    Truck: 0.7469
    Van: 0.5679
    mAP: 0.5930
