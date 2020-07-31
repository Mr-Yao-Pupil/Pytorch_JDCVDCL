from torchvision import transforms

# txt文件的根路径，所有标签txt文件全部存放于同一个路径
ANNO_DIR = "data_img"

# 图片的transforms处理,根据项目需求自行更改
DATA_TRANSFORMS = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

premodulepath = r"/home/ubuntu/Yaozhichao/Litong_JDCV_DCL/weight/resnet50-19c8e357.pth"