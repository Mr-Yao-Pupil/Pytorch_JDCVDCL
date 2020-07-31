from DCL_Dataset import MyDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
from util.DataLoad_Mode import *
from torch.autograd import Variable
import cfg
from util.DataLoad_Mode import collate_fn4train, collate_fn4val
from DCL_Net import MainNet

module_savepath = r"/home/ubuntu/Yaozhichao/Litong_JDCV_DCL/weight"

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet(4, mode="train")
    net = nn.DataParallel(net)

    savepath = os.path.join(module_savepath, "28_300.pth")
    if os.path.isfile(savepath):
        net.load_state_dict(torch.load(savepath))
        print("模型加载完成")

    else:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(premodulepath)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print("预训练模型加载完成")

    net = net.to(device)

    opt = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 80], gamma=0.1)

    epoch = 1
    train_datasets = MyDataset(anno_dir=cfg.ANNO_DIR,
                               mode="train",
                               cls_num=4,
                               swap_size=[7, 7],
                               data_transforms=cfg.DATA_TRANSFORMS)
    train_loader = DataLoader(train_datasets,
                              batch_size=32,
                              shuffle=True,
                              drop_last=True,
                              num_workers=32,
                              pin_memory=True,
                              collate_fn=collate_fn4train)

    test_datasets = MyDataset(anno_dir=cfg.ANNO_DIR,
                              mode="val",
                              cls_num=4,
                              swap_size=[7, 7],
                              data_transforms=cfg.DATA_TRANSFORMS)
    test_loader = DataLoader(test_datasets,
                             batch_size=8,
                             shuffle=True,
                             drop_last=True,
                             num_workers=8,
                             pin_memory=True,
                             collate_fn=collate_fn4val)

    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.CrossEntropyLoss()
    loss_fn3 = nn.L1Loss()

    print("数据集加载完成")
    while True:
        for i, (img_data, label, label_swap, law_swap) in enumerate(train_loader):
            inputs = Variable(img_data.cuda())
            labels = Variable(torch.from_numpy(np.array(label)).cuda())
            labels_swap = Variable(torch.from_numpy(np.array(label_swap)).cuda())
            swap_law = Variable(torch.from_numpy(np.array(law_swap)).float().cuda())

            outputs = net(inputs)

            loss1 = loss_fn1(outputs[0], labels)
            loss2 = loss_fn2(outputs[1], labels_swap)
            loss3 = loss_fn3(outputs[2], swap_law)

            loss = loss1 + loss2 + loss3 * 0.01
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"{epoch}-{i}:Loss:{loss}===>loss1:{loss1}===>loss2:{loss2}===>loss3:{loss3}")

            if i % 100 == 0:
                net.eval()
                with torch.no_grad():
                    acc_loss = 0
                    acc = 0
                    for img_data, label, label_swap, law_swap in test_loader:
                        inputs = Variable(img_data.cuda())
                        labels = Variable(torch.from_numpy(np.array(label)).cpu())
                        labels_swap = Variable(torch.from_numpy(np.array(label_swap)).cpu())
                        swap_law = Variable(torch.from_numpy(np.array(law_swap)).float().cpu())

                        output1, output2, output3 = net(inputs)
                        output1, output2, output3 = output1.cpu(), output2.cpu(), output3.cpu()

                        loss1 = loss_fn1(output1, labels)
                        loss2 = loss_fn2(output2, labels_swap)
                        loss3 = loss_fn3(output3, swap_law)
                        loss = loss1 + loss2 + loss3
                        acc_loss += loss

                        logit = torch.argmax(output1, dim=1)

                        acc1 = (logit.cpu() == labels.cpu()).sum()
                        acc += acc1.item()
                    print("准确度：", acc / len(test_datasets))
                    print("平均损失：", acc_loss / 25)

                    if acc / len(test_datasets) >= 0.88:
                        save_path = os.path.join(module_savepath, f"{epoch}_{i}.pth")
                        torch.save(net.state_dict(), save_path)
            net.train()
        epoch += 1
        scheduler.step()
