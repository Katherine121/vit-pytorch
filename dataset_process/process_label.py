import decimal
import math
import random
import numpy as np
import torch
import os
import shutil

torch.set_printoptions(precision=15)


def get_pics(datapath):
    # process pics
    pics_list = list()
    file_path = os.path.join(datapath, "images.0_DownLeft_2")
    pics_path = os.listdir(file_path)

    for i in range(0, len(pics_path)):
        full_path = os.path.join(file_path, pics_path[i])
        pics_list.append(full_path)
    list.sort(pics_list)

    # process coordinates
    labels = []
    f = open(os.path.join(datapath, "pos_xy.txt"), 'rt')
    for line in f:
        line = line.replace('\n', '')
        labels.append(list(map(eval, line.split(','))))
    f.close()

    return pics_list, labels


if __name__ == "__main__":
    paths = ['../images/62576', '../images/62577', '../images/62748', '../images/62750']
    k = 400
    epoch = 100

    pics_list = []
    labels = []
    for i in range(0, 4):
        result = get_pics(paths[i])
        pics_list += result[0]
        labels += result[1]

    # 随机生成k个初始聚类中心，保存为centre
    centre = np.empty((k, 2))
    for i in range(0, k):
        index = random.randint(0, len(labels) - 1)
        centre[i][0] = labels[index][0]
        centre[i][1] = labels[index][1]

    # 迭代epoch次
    for iter in range(0, epoch):
        print(iter)
        # 计算欧氏距离
        def euclidean_distance(pos1, pos2):
            return math.sqrt(((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))

        # 每个点到每个中心点的距离矩阵
        dis = np.empty((len(labels), k))
        for i in range(0, len(labels)):
            for j in range(0, k):
                dis[i][j] = euclidean_distance(labels[i], centre[j])

        # 初始化分类矩阵
        classify = []
        for i in range(0, k):
            classify.append([])

        # 比较距离并重新分成k类
        for i in range(0, len(labels)):
            List = dis[i].tolist()
            index = List.index(dis[i].min())
            # classify是从小到大添加的
            classify[index].append(i)

        # 构造新的中心点
        new_centre = np.empty((k, 2))
        for index in range(0, len(classify)):
            x_sum = 0
            y_sum = 0
            # 避免缺失簇
            if len(classify[index]) == 0:
                randindex = random.randint(0, len(labels) - 1)
                new_centre[index][0] = labels[randindex][0]
                new_centre[index][1] = labels[randindex][1]
                continue

            for labelindex in range(0, len(classify[index])):
                x_sum += labels[classify[index][labelindex]][0]
                y_sum += labels[classify[index][labelindex]][1]

            new_centre[index][0] = x_sum / len(classify[index])
            new_centre[index][1] = y_sum / len(classify[index])

        # 比较新的中心点和旧的中心点是否一样
        if (new_centre == centre).all():
            break
        else:
            centre = new_centre

    for index in range(0, k):
        # 创建聚类文件夹
        if os.path.exists("../label_images") is False:
            os.mkdir("../label_images")
        if os.path.exists("../label_pos") is False:
            os.mkdir("../label_pos")

        new_image_path = "../label_images" + "/" + "class" + str(index)
        if os.path.exists(new_image_path) is False:
            os.mkdir(new_image_path)
        new_pos_path = "../label_pos" + "/" + "class" + str(index)
        if os.path.exists(new_pos_path) is False:
            os.mkdir(new_pos_path)

        # 记录簇中心
        with open(new_pos_path + "/" + "center.txt", "a") as file1:
            file1.write(str(new_centre[index][0]) + "," + str(new_centre[index][1]) + "\n")
        file1.close()

        # 因为四个文件夹中的照片正好是从小到大，
        # 所以按照get_pics的索引，移动到新文件夹，
        # 在获取数据集时排序，和原来的就一模一样了
        # 如果四个文件夹的照片不是从小到大，就不需要在获取数据集的时候排序？否则打乱了原本加入到新文件夹中的顺序
        for labelindex in range(0, len(classify[index])):
            # 创建每簇的坐标文件
            class_x = labels[classify[index][labelindex]][0]
            class_y = labels[classify[index][labelindex]][1]
            with open(new_pos_path + "/" + "pos_xy.txt", "a") as file1:
                file1.write(str(class_x) + "," + str(class_y) + "\n")
            file1.close()

            # 移动每簇的图片
            ori_path = pics_list[classify[index][labelindex]]
            # print(ori_path)
            shutil.copy(ori_path, new_image_path)


    print('迭代次数为：', iter + 1)
    print('聚类中心为：', new_centre)
    # print('分类情况为：', classify)

