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


y_pred = list(np.random.uniform(0.4, 0.6, 2000)) + list(np.random.uniform(0.5, 0.7, 8000))
y_true = [0] * 2000 + [1] * 8000

print(AUC(y_true, y_pred))