import numpy as np


def PRF1(data_list, thd):
    data = np.array(data_list)

    # 将>= thd 设置为1，其他设置为0
    pre = data[:, 1]
    pre[np.where(data[:, 1] >= thd)] = 1
    pre[np.where(data[:, 1] < thd)] = 0

    y_true, y_pre = data[:, 2], data[:, 1]

    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pre, 1)))
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pre, 1)))
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pre, 0)))
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pre, 0)))

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)

    print("精确率为:", Precision)
    print("召回率为:", Recall)
    print("总体精度为:", Accuracy)
    print("F1分数为:", F1_score)


data = [[0, 0.76, 1], [1, 0.3, 0], [2, 0.5, 0], [3, 0.76, 1]]
print(PRF1(data, 0.5))