import numpy as np
import random
import collections


def distance(point1, point2):  # 计算距离（欧几里得距离）
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_means(data, k, max_iter=1000):
    centers = {}  # 初始聚类中心
    # 初始化，选k个样本作为初始聚类中心
    # random.sample(): 随机不重复抽取k个值
    for idx, i in enumerate(random.sample(range(data.shape[0]), k)):
        # idx取值范围[0, k-1]，代表第几个聚类中心;  i为随机选取的样本作为聚类中心的编号
        centers[idx] = data[i]

    for i in range(max_iter):
        print("第{}次迭代".format(i + 1))
        clusters = collections.defaultdict(list)  # 聚类结果，聚类中心的 索引idx -> [样本集合]

        for row in data:  # 遍历每个样本
            distances = []  # 计算该样本到每个聚类中心的距离 (只会有k个元素)
            for c in centers:  # 遍历每个聚类中心
                distances.append(distance(row, centers[c]))
            idx = np.argmin(distances)  # 最小距离的索引
            clusters[idx].append(row)  # 将该样本添加到第idx个聚类中心

        pre_centers = centers.copy()  # 记录之前的聚类中心点

        for c in clusters.keys():
            # 重新计算中心点（计算该聚类中心的所有样本的均值）
            centers[c] = np.mean(clusters[c], axis=0)

        optimizer = True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                optimizer = False
                break

        if optimizer == True:
            print("新旧聚类中心不变，迭代停止")
            break
    return centers


def predict(p_data, centers):
    distances = [distance(p_data, centers[c]) for c in centers]
    return np.argmin(distances)


x = np.random.randint(0, 10, size=(200, 2))
centers = k_means(x, 3)
print(centers)
p_data = [4, 7]
print("{}应该属于第{}个聚类中心".format(p_data, predict(p_data, centers)))

# def distance(point1, point2):
#     return np.sqrt(np.sum((point1 - point2) ** 2))
#
#
# def k_means(data, k, max_iter=1000):
#     centers = {}
#     for idx, i in enumerate(random.sample(range(data.shape[0]), k)):
#         centers[idx] = data[i]
#     for i in range(max_iter):
#         print("第{}次迭代".format(i+1))
#         clusters = collections.defaultdict(list)
#         for row in data:
#             distances = []
#             for c in centers:
#                 distances.append(distance(row, centers[c]))
#             min_idx = np.argmin(distances)
#             clusters[min_idx].append(row)
#         pre_centers = centers.copy()
#         for c in clusters.keys():
#             centers[c] = np.mean(clusters[c], axis=0)
#         optimizer = True
#         for c in centers:
#             if distance(centers[c], pre_centers[c]) > 1e-8:
#                 optimizer = False
#                 break
#         if optimizer:
#             print("新旧聚类中心点不变，迭代停止")
#             break
#     return centers



