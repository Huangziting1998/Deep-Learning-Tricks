import numpy as np
import pandas as pd


def AUC(y_true, y_pred):
    pair = list(zip(y_true, y_pred))
    pair = sorted(pair, key=lambda x: x[1])  # 进行排序
    df = pd.DataFrame([[x[0], x[1], i + 1] for i, x in enumerate(pair)], columns=['y_true', 'y_pred', 'rank'])

    # 下面为预测值一样的序号进行重新编号
    for k, v in df.y_pred.value_counts().items():
        if v == 1:  # 预测值k只出现了一次，continue
            continue
        rank_mean = df[df.y_pred == k]['rank'].mean()
        df.loc[df.y_pred == k, 'rank'] = rank_mean

    pos_df = df[df.y_true == 1]  # 正样本的表
    m = pos_df.shape[0]  # 正样本数
    n = df.shape[0] - m  # 负样本数
    return (pos_df['rank'].sum() - m * (m + 1) / 2) / (m * n)


def get_roc(y_pred, y_true):
    pos, neg = y_true[y_true == 1], y_true[y_true == 0]
    # 按概率大小逆序排列
    rank = np.argsort(y_pred)[::-1]
    y_pred, y_true = y_pred[rank], y_true[rank]
    tpr_all, fpr_all = [0], [0]
    tpr, fpr = 0, 0
    x_step, y_step = 1 / float(len(neg)), 1 / float(len(pos))
    y_sum = 0  # 用于计算AUC
    for i in range(len(y_pred)):
        if y_true[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr

    return tpr_all, fpr_all, y_sum * x_step  # 获得总体TPR，FPR和相应的AUC


y_pred = np.array(list(np.random.uniform(0.4, 0.6, 2000)) + list(np.random.uniform(0.5, 0.7, 8000)))
y_true = np.array([0] * 2000 + [1] * 8000)
#########################################################################################################
print(AUC(y_true, y_pred))
#########################################################################################################
tpr_all, fpr_all, auc = get_roc(y_pred, y_true)
print(len(tpr_all) == len(fpr_all))
print(auc)
