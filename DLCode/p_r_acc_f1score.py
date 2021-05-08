import numpy as np


def p_r_f1(data, thd):
    data = np.array(data)

    # 将>= thd 设置为1，其他设置为0
    pred = data[:, 1]
    pred[np.where(data[:, 1] >= thd)] = 1
    pred[np.where(data[:, 1] < thd)] = 0

    y_true = data[:, 2]
    y_pred = pred

    tp = np.sum(np.logical_and(np.equal(1, y_true), np.equal(1, y_pred)))
    fp = np.sum(np.logical_and(np.equal(0, y_true), np.equal(1, y_pred)))
    fn = np.sum(np.logical_and(np.equal(1, y_true), np.equal(0, y_pred)))
    tn = np.sum(np.logical_and(np.equal(0, y_true), np.equal(0, y_pred)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("precision", precision)
    print("recall", recall)
    print("accuracy", accuracy)
    print("f1_score", f1_score)


data = [[0, 0.76, 1], [1, 0.3, 0], [2, 0.5, 0], [3, 0.76, 1]]
p_r_f1(data, 0.5)