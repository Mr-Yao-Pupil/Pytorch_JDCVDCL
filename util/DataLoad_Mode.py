import torch


def collate_fn4train(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])
        if sample[3] == -1:
            label_swap.append(1)
            label_swap.append(0)
        else:
            label_swap.append(sample[2])
            label_swap.append(sample[3])
        law_swap.append(sample[4])
        law_swap.append(sample[5])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap


def collate_fn4val(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        if sample[3] == -1:
            label_swap.append(1)
        else:
            label_swap.append(sample[2])
        law_swap.append(sample[3])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap


def collate_fn4backbone(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        if len(sample) == 7:
            label.append(sample[2])
        else:
            label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


def collate_fn4test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


def testloader(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
    return torch.stack(imgs, 0), label