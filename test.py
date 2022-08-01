import os

import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from data_augment import autoaugment
from data_augment.utils import cutmix, cutmix_criterion
from dataset_process.dataloader import TrainDataset, TestDataset, get_center

from vit_pytorch.mobile_vit import MobileViT_S, MobileViT_XXS


def check_accuracy(loader, model, device=None, dtype=None):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_diff = 0

    with torch.no_grad():
        t = 0
        for x, y, pos in loader:
            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=torch.long)
            pos = pos.to(device, dtype=dtype)

            outputs = model(x)

            # _,是batch_size*概率，preds是batch_size*最大概率的列号
            _, preds = outputs.max(1)

            num_correct = (preds == y).sum()
            num_samples = preds.size(0)

            total_correct += num_correct
            total_samples += num_samples

            pos_outputs = get_center(preds).to(device)
            pos_diff = nn.functional.pairwise_distance(pos, pos_outputs).sum().item() / num_samples

            total_diff += pos_diff

            # 每200个iteration打印一次测试集准确率
            if t % 200 == 0:
                print('预测正确的图片数目' + str(num_correct))
                print('总共的图片数目' + str(num_samples))
                print('预测坐标与真实坐标的平均欧式距离' + str(pos_diff))

            t += 1

        acc = float(total_correct) / total_samples
        diff = float(total_diff) / total_samples
    return acc


def train(
        loader_train=None, loader_val=None,
        device=None, dtype=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None, optimizer=None,
        epochs=300, check_point_dir=None, save_epochs=None
):
    acc = 0
    accs = [0]
    losses = []
    best_acc = 0

    for e in range(epochs):
        model.train()
        total_loss = 0
        for t, (x, y, _) in enumerate(loader_train):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            # 原x+混x，原t，混t，原混比
            inputs, targets_a, targets_b, lam = cutmix(x, y, 1)
            # 原x+混x->原y+混y
            outputs = model(inputs)

            # 原y+混y和原t，混t求损失：lam越大，小方块越小，被识别成真图片的概率越大
            # 2
            loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss_value = np.array(loss.item())
            total_loss += loss_value

            # 1
            optimizer.zero_grad()
            # 3
            loss.backward()
            # 4
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # 200个iteration打印一下训练集损失
            if t % 200 == 0:
                print("Iteration:" + str(t) + ', average Loss = ' + str(loss_value))

        total_loss /= t
        losses.append(total_loss)

        acc = check_accuracy(loader_val, model, device=device)
        accs.append(np.array(acc))

        # 每个epoch记录一次测试集准确率和所有batch的平均训练损失
        print("Epoch:" + str(e) + ', Val acc = ' + str(acc) + ', average Loss = ' + str(total_loss))

        if os.path.exists(check_point_dir) is False:
            os.mkdir(check_point_dir)

        # 将每个epoch的平均损失写入文件
        with open(check_point_dir + "/" + "avgloss.txt", "a") as file1:
            file1.write(str(total_loss) + '\n')
        file1.close()
        # 将每个epoch的测试集准确率写入文件
        with open(check_point_dir + "/" + "testacc.txt", "a") as file2:
            file2.write(str(acc) + '\n')
        file2.close()

        # 如果到了保存的epoch或者是训练完成的最后一个epoch
        if acc > best_acc:
            best_acc = acc
            model.eval()
            # 保存模型参数
            torch.save(model.state_dict(), check_point_dir + "/" + "model.pth")
            # 保存模型结构
            torch.save(model, check_point_dir + "/" + "model.pt")

    return acc


def run(
        loader_train=None, loader_val=None,
        device=None, dtype=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        T_mult=2,
        epoch=300, lr=0.0009, wd=0.10,
        check_point_dir=None, save_epochs=3,

):
    epochs = epoch
    model_ = model
    learning_rate = lr
    weight_decay = wd
    print('Training under lr: ' + str(lr) + ' , wd: ' + str(wd) + ' for ', str(epochs), ' epochs.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=T_mult)

    args = {
        'loader_train': loader_train, 'loader_val': loader_val,
        'device': device, 'dtype': dtype,
        'model': model_,
        'criterion': criterion,
        'scheduler': lr_scheduler, 'optimizer': optimizer,
        'epochs': epochs,
        'check_point_dir': check_point_dir, 'save_epochs': save_epochs,
    }

    print('#############################     Training...     #############################')

    val_acc = train(**args)

    # 最后一个epoch的最后一次测试集准确率
    print('Training for ' + str(epochs) + ' epochs, learning rate: ', learning_rate,
          ', weight decay: ', weight_decay, ', Val acc: ', val_acc)
    print('Done, model saved in ', check_point_dir)


def load_mobilevit_weights(model_path):
    # 1加载模型结构
    net = MobileViT_XXS(img_size=256, num_classes=1000)

    # 2加载模型权重
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']

    # Since there is a problem in the names of layers, we will change the keys to meet the MobileViT model architecture
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)

    # Once the keys are fixed, we can modify the parameters of MobileViT
    net.load_state_dict(state_dict)

    return net


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_aug = transforms.Compose([
        autoaugment.CIFAR10Policy(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_crop = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 原图
    trainDataset = TrainDataset(transform=transform, path1="label_images", path2="label_pos", class_num=400)
    # 自动增强1
    augtrain1 = TrainDataset(transform=transform_aug, path1="label_images", path2="label_pos", class_num=400)
    # 自动增强2
    augtrain2 = TrainDataset(transform=transform_aug, path1="label_images", path2="label_pos", class_num=400)
    # 随机裁剪
    augtrain3 = TrainDataset(transform=transform_crop, path1="label_images", path2="label_pos", class_num=400)
    trainLoader = DataLoader(trainDataset + augtrain1 + augtrain2 + augtrain3,
                             batch_size=64, shuffle=True, drop_last=False)

    print(len(trainDataset))
    print(len(augtrain1))
    print(len(augtrain2))
    print(len(augtrain3))

    # 原图
    testDataset = TestDataset(transform=transform, path1="label_images", path2="label_pos", class_num=400)
    # 自动增强1
    augtest1 = TestDataset(transform=transform_aug, path1="label_images", path2="label_pos", class_num=400)
    # 自动增强2
    augtest2 = TestDataset(transform=transform_aug, path1="label_images", path2="label_pos", class_num=400)
    # 随机裁剪
    augtest3 = TestDataset(transform=transform_crop, path1="label_images", path2="label_pos", class_num=400)
    testLoader = DataLoader(testDataset + augtest1 + augtest2 + augtest3,
                            batch_size=16, shuffle=True, drop_last=False)

    print(len(testDataset))
    print(len(augtest1))
    print(len(augtest2))
    print(len(augtest3))

    print('###############################  Dataset loaded  ##############################')

    print('############################### Model loading ###############################')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    # 1加载模型结构
    # 2加载模型权重
    model = load_mobilevit_weights("xxsmodel_best.pth.tar")

    # 修改模型输出
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_features=in_feature, out_features=400)

    # 3设置运行环境
    model = model.to(device)

    print('###############################  Model loaded  ##############################')

    args = {
        'loader_train': trainLoader, 'loader_val': testLoader,
        'device': device, 'dtype': torch.float32,
        'model': model,
        'criterion': nn.CrossEntropyLoss(),
        # 余弦退火
        'T_mult': 2,
        'epoch': 300, 'lr': 0.0009, 'wd': 0.10,
        'check_point_dir': "saved_model2", 'save_epochs': 3,
    }
    run(**args)
