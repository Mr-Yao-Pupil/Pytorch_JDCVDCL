from DCL_Dataset import MyDataset
from torch.utils.data import DataLoader
from DCL_Net import MainNet
import os
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import cfg
from util.DataLoad_Mode import *

if __name__ == '__main__':
    module_savepath = "/home/ubuntu/Yaozhichao/Litong_JDCV_DCL/weight/18_0.pth"
    # os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    test_datasets = MyDataset(anno_dir=cfg.ANNO_DIR,
                              mode="test",
                              cls_num=4,
                              swap_size=[7, 7],
                              data_transforms=cfg.DATA_TRANSFORMS)
    test_loader = DataLoader(test_datasets,
                             batch_size=8,
                             shuffle=True,
                             drop_last=True,
                             num_workers=8,
                             pin_memory=True,
                             collate_fn=testloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet(4, mode="test")

    if os.path.isfile(module_savepath):
        # net.load_state_dict(torch.load(module_savepath))
        model_dict = net.state_dict()

        pretrained_dict = torch.load(module_savepath)

        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}

        model_dict.update(pretrained_dict)

        net.load_state_dict(model_dict)
        print("模型加载完成")

    # net.cuda()
    net = net.to(device)

    net = nn.DataParallel(net)
    net.eval()
    acc = 0
    with torch.no_grad():
        for img_data, label in test_loader:
            inputs = Variable(img_data.cuda())
            labels = Variable(torch.from_numpy(np.array(label)).cpu())

            output1 = net(inputs)
            output1 = output1.cpu()

            logit = torch.argmax(output1, dim=1)

            acc1 = (logit.cpu() == labels.cpu()).sum()
            acc += acc1.item()
            print(logit)
            print(labels)
        print("准确度：", acc / len(test_datasets))