import numpy as np


def split_train_test(data, test_ratio):
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(1222)
    # permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))
    # test_ratio为测试集所占的比例
    test_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    return data[train_indices], data[test_indices]


# 测试
data = np.array(np.random.uniform(0.4, 0.6, 2000))
train_set, test_set = split_train_test(data, 0.2)
print(len(train_set), "train +", len(test_set), "test")
