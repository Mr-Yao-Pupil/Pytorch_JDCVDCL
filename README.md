### 实现的内容

-  主干特征提取网络：目前只调用了ResNet50
-  特征金字塔：SPP，PAN
-  训练用到的小技巧：学习率余弦退火衰减
-  激活函数：作者的其他函数暂未复现使用

### 编码环境

torch==1.5.0

### 注意事项

还没想到

### 小技巧的设置

1. 中途终止训练直接停止代码，导入已经训练好的模更改与训练文件的地址即可

### 文件下载

在执行时会自动从官方的预训练的模型下载参数

### 预测步骤

#### 1、使用预训练权重

1. 在cfg.py文件中更改预训练权重的本地路径，具体参看pytorch官方文档路径
2. 将所有的图片按照分类放入文件夹：data_img
3. 通过Generate_TXT.py生成标签文件

#### 2、使用自己训练的权重

a、按照训练步骤训练。
b、在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。

```
_defaults = {
    "model_path": 'model_data/yolo4_weights.pth',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt',
    "model_image_size" : (416, 416, 3),
    "confidence": 0.5,
    "cuda": True
}
```

c、运行predict.py，输入

```
img/street.jpg
```

可完成预测。
d、利用video.py可进行摄像头检测。

### 训练步骤

1、本文使用VOC格式进行训练。
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
4、在训练前利用voc2yolo4.py文件生成对应的txt。
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**

```
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```

6、此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。
7、**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：

```
classes_path = 'model_data/new_classes.txt'    
```

model_data/new_classes.txt文件内容为：

```
cat
dog
...
```

8、运行train.py即可开始训练。