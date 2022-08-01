import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
import os

from torchvision import transforms

torch.set_printoptions(precision=15)


def get_center(preds, path="label_pos"):
    b = preds.size(0)

    center = None
    for i in range(0, b):
        # 坐标文件夹
        class_path = os.path.join(path, "class" + str(preds[i].item()))

        # 加载坐标
        f = open(os.path.join(class_path, "center.txt"), 'rt')
        for line in f:
            line = line.replace('\n', '')
            cur = torch.tensor(list(map(eval, line.split(','))), dtype=torch.float32).unsqueeze(dim=0)
            if center is None:
                center = cur
            else:
                center = torch.cat((center, cur), dim=0)
        f.close()

    return center


class TrainDataset(Dataset):
    def __init__(self, transform, path1="label_images", path2="label_pos", class_num=400):
        self.transform = transform

        res = []
        for i in range(0, class_num):
            # 图片文件夹
            class_list = []
            class_path1 = os.path.join(path1, "class" + str(i))
            pics_path = os.listdir(class_path1)

            # 加载图片
            for j in range(0, len(pics_path)):
                full_path = os.path.join(class_path1, pics_path[j])
                class_list.append(full_path)
            list.sort(class_list)

            # 坐标文件夹
            class_path2 = os.path.join(path2, "class" + str(i))

            # 加载坐标
            labels = []
            f = open(os.path.join(class_path2, "pos_xy.txt"), 'rt')
            for line in f:
                line = line.replace('\n', '')
                labels.append(list(map(eval, line.split(','))))
            f.close()

            # 加入图片，类别标签，坐标，只取前4/5的图片作为训练集
            for j in range(0, (int)(len(pics_path) * 0.8)):
                res.append((class_list[j], i, labels[j]))

        # print(len(res))

        self.imgs = res

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic, label, pos = self.imgs[index]

        pic = Image.open(pic)
        pic = pic.convert('RGB')
        pic = self.transform(pic)

        pos = torch.tensor(pos, dtype=torch.float32)

        return pic, label, pos


class TestDataset(Dataset):
    def __init__(self, transform, path1="label_images", path2="label_pos", class_num=400):
        self.transform = transform

        res = []
        for i in range(0, class_num):
            # 图片文件夹
            class_list = []
            class_path1 = os.path.join(path1, "class" + str(i))
            pics_path = os.listdir(class_path1)

            # 加载图片
            for j in range(0, len(pics_path)):
                full_path = os.path.join(class_path1, pics_path[j])
                class_list.append(full_path)
            list.sort(class_list)

            # 坐标文件夹
            class_path2 = os.path.join(path2, "class" + str(i))

            # 加载坐标
            labels = []
            f = open(os.path.join(class_path2, "pos_xy.txt"), 'rt')
            for line in f:
                line = line.replace('\n', '')
                labels.append(list(map(eval, line.split(','))))
            f.close()

            # 加入图片，类别标签，坐标，只取前4/5的图片作为训练集
            for j in range((int)(len(pics_path) * 0.8), len(pics_path)):
                res.append((class_list[j], i, labels[j]))

        # print(len(res))

        self.imgs = res

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic, label, pos = self.imgs[index]

        pic = Image.open(pic)
        pic = pic.convert('RGB')
        pic = self.transform(pic)

        pos = torch.tensor(pos, dtype=torch.float32)

        return pic, label, pos


if __name__ == "__main__":
    path1 = '../images/62576'
    path2 = '../images/62577'
    path3 = '../images/62748'
    path4 = '../images/62750'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainDataset = TrainDataset(transform=transform, path1="../label_images", path2="../label_pos", class_num=400)
    testDataset = TestDataset(transform=transform, path1="../label_images", path2="../label_pos", class_num=400)
    print(len(trainDataset) + len(testDataset))
