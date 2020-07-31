import torch
from torch.utils.data import Dataset, DataLoader
import os
import cfg
from PIL import Image, ImageStat
import random
from util.DataLoad_Mode import *


class MyDataset(Dataset):
    def __init__(self, anno_dir, mode, cls_num, swap_size=[7, 7], data_transforms=cfg.DATA_TRANSFORMS):
        self.mode = mode
        self.anno_dir = anno_dir
        self.cls_num = cls_num
        self.swap_size = swap_size
        self.data_transforms = data_transforms

        assert self.mode == "train" or self.mode == "val" or self.mode == "test"

        # 可更改数据集txt名称
        if self.mode == "train":
            self.anno_path = os.path.join(self.anno_dir, "Train.txt")
            self.dataset = open(self.anno_path).readlines()

        if self.mode == "val":
            self.anno_path = os.path.join(self.anno_dir, "Test.txt")
            self.dataset = open(self.anno_path).readlines()

        if self.mode == "test":
            self.anno_path = os.path.join(self.anno_dir, "Val.txt")
            self.dataset = open(self.anno_path).readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        line = self.dataset[index].split(" ")

        img_path = line[0]
        img = Image.open(img_path)

        if self.mode == "train" or self.mode == "val":
            img_unswap = img
            img_unswap_list = self.crop_img(img_unswap, self.swap_size)

            swap_range = self.swap_size[0] * self.swap_size[1]

            law_unswap = [(i - swap_range // 2) / swap_range for i in range(swap_range)]

            img_swap = self.swap(img_unswap, self.swap_size)
            img_swap_list = self.crop_img(img_swap, self.swap_size)

            unswap_stats = [sum(ImageStat.Stat(im).mean) for im in img_unswap_list]
            swap_stats = [sum(ImageStat.Stat(im).mean) for im in img_swap_list]

            law_swap = []
            for im_swap in swap_stats:
                distance = [abs(im_swap - im_unswap) for im_unswap in unswap_stats]
                i = distance.index(min(distance))
                law_swap.append((i - (swap_range // 2)) / swap_range)

            img_swap = self.data_transforms(img_swap)
            img_unswap = self.data_transforms(img_unswap)

            label = int(line[1])
            if self.cls_num > 2:
                label_swap = label + self.cls_num
            else:
                label_swap = -1

            if self.mode == "train":
                return img_unswap, img_swap, label, label_swap, law_unswap, law_swap
            elif self.mode == "val":
                law_swap = [(i - (swap_range // 2)) / swap_range for i in range(swap_range)]
                label_swap = label
                return img_unswap, label, label_swap, law_unswap, law_swap

        elif self.mode == "test":
            img = self.data_transforms(img)
            label = int(line[1])

            return img, label

    def crop_img(self, image, crop_size):
        width, heigh = image.size

        # 计算等分点
        crop_x = [int(width / crop_size[0] * i) for i in range(crop_size[0] + 1)]
        crop_y = [int(heigh / crop_size[1] * i) for i in range(crop_size[1] + 1)]

        img_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                img_list.append(
                    image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], heigh))))

        return img_list

    def swap(self, img, crop):
        def crop_image(image, cropnum):
            width, high = image.size
            crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
            crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
            im_list = []
            for j in range(len(crop_y) - 1):
                for i in range(len(crop_x) - 1):
                    im_list.append(
                        image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
            return im_list

        widthcut, highcut = img.size
        img = img.crop((10, 10, widthcut - 10, highcut - 10))
        images = crop_image(img, crop)
        pro = 5
        if pro >= 5:
            tmpx = []
            tmpy = []
            count_x = 0
            count_y = 0
            k = 1
            RAN = 2
            for i in range(crop[1] * crop[0]):
                tmpx.append(images[i])
                count_x += 1
                if len(tmpx) >= k:
                    tmp = tmpx[count_x - RAN:count_x]
                    random.shuffle(tmp)
                    tmpx[count_x - RAN:count_x] = tmp
                if count_x == crop[0]:
                    tmpy.append(tmpx)
                    count_x = 0
                    count_y += 1
                    tmpx = []
                if len(tmpy) >= k:
                    tmp2 = tmpy[count_y - RAN:count_y]
                    random.shuffle(tmp2)
                    tmpy[count_y - RAN:count_y] = tmp2
            random_im = []
            for line in tmpy:
                random_im.extend(line)

            # random.shuffle(images)
            width, high = img.size
            iw = int(width / crop[0])
            ih = int(high / crop[1])
            toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
            x = 0
            y = 0
            for i in random_im:
                i = i.resize((iw, ih), Image.ANTIALIAS)
                toImage.paste(i, (x * iw, y * ih))
                x += 1
                if x == crop[0]:
                    x = 0
                    y += 1
        else:
            toImage = img
        toImage = toImage.resize((widthcut, highcut))
        return toImage


if __name__ == '__main__':
    from utils.LoadMode import *
    from torch.autograd import Variable
    import numpy as np

    """
    训练数据集导入测试
    """
    # datasets = MyDataset(anno_dir=cfg.ANNO_DIR,
    #                      mode="train",
    #                      cls_num=4,
    #                      swap_size=[7, 7],
    #                      data_transforms=cfg.DATA_TRANSFORMS)
    #
    # dataloader = DataLoader(datasets,
    #                         batch_size=2,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         num_workers=2,
    #                         pin_memory=True,
    #                         collate_fn=collate_fn4train)
    #
    # for data in dataloader:
    #     inputs = Variable(data[0].cuda())
    #     labels = Variable(torch.from_numpy(np.array(data[1])).cuda())
    #     labels_swap = Variable(torch.from_numpy(np.array(data[2])).cuda())
    #     swap_law = Variable(torch.from_numpy(np.array(data[3])).float().cuda())
    #
    #     print(inputs.shape)
    #     print(labels.shape)
    #     print(labels.shape)
    #     print(swap_law.shape)

    """
    验证数据集导入测试
    """
    # datasets = MyDataset(anno_dir=cfg.ANNO_DIR,
    #                      mode="val",
    #                      cls_num=4,
    #                      swap_size=[7, 7],
    #                      data_transforms=cfg.DATA_TRANSFORMS)
    #
    # dataloader = DataLoader(datasets,
    #                         batch_size=2,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         num_workers=2,
    #                         pin_memory=True,
    #                         collate_fn=collate_fn4val)
    #
    # for data in dataloader:
    #     inputs = Variable(data[0].cuda())
    #     labels = Variable(torch.from_numpy(np.array(data[1])).cuda())
    #     labels_swap = Variable(torch.from_numpy(np.array(data[2])).cuda())
    #     swap_law = Variable(torch.from_numpy(np.array(data[3])).float().cuda())
    #
    #     print(inputs.shape)
    #     print(labels.shape)
    #     print(labels.shape)
    #     print(swap_law.shape)
    #     exit()

    """
    测试数据集导入测试
    """
    # datasets = MyDataset(anno_dir=cfg.ANNO_DIR,
    #                      mode="test",
    #                      cls_num=4,
    #                      swap_size=[7, 7],
    #                      data_transforms=cfg.DATA_TRANSFORMS)
    #
    # dataloader = DataLoader(datasets,
    #                         batch_size=2,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         num_workers=2,
    #                         pin_memory=True,
    #                         collate_fn=testloader)
    #
    # for data in dataloader:
    #     inputs = Variable(data[0].cuda())
    #     labels = Variable(torch.from_numpy(np.array(data[1])).cuda())
    #
    #     print(inputs.shape)
    #     print(labels.shape)