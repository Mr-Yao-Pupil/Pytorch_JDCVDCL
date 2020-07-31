import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

net_module = {'resnet50': torchvision.models.resnet50()}


class MainNet(nn.Module):
    def __init__(self, sumcls, mode):
        super(MainNet, self).__init__()
        self.sumcls = sumcls
        self.mode = mode

        assert self.mode == "train" or self.mode == "val" or self.mode == "test"

        self.model = net_module['resnet50']
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.adaavgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Linear(2048, self.sumcls, bias=False)

        if self.mode == "train" or self.mode == "val":
            self.classifier_swap = nn.Linear(2048, 2 * self.sumcls, bias=False)

            self.Convmask = nn.Conv2d(2048, 1, 1, 1, 0)

            self.region_pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        main_feature = self.model(x)
        feature_vector = self.adaavgpool(main_feature).reshape(x.size(0), -1)
        arc_out = self.classifier(feature_vector)

        assert self.mode == "train" or self.mode == "val" or self.mode == "test"

        if self.mode == "train" or self.mode == "val":
            mask = self.Convmask(main_feature)
            mask = self.region_pool(mask)
            mask = torch.tanh(mask)
            mask = mask.reshape(mask.shape[0], -1)

            ada_out = self.classifier_swap(feature_vector)

            return arc_out, ada_out, mask
        elif self.mode == "test":
            return arc_out


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    net = MainNet(4, mode="train").cuda()
    net = nn.DataParallel(net)
    test_tensor = torch.Tensor(5, 3, 448, 448).cuda()

    out = net(test_tensor)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)