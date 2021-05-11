import numpy as np


def nms(dets, thresh):
    # dets某个类的框，x1、y1、x2、y2、以及置信度score
    # eg:dets为[[x1,y1,x2,y2,score],[x1,y1,y2,score]……]]
    # thresh是IoU的阈值
    x1, y1 = dets[:, 0], dets[:, 1]
    x2, y2 = dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score置信度降序排序
    order = np.argsort(scores)[::-1]
    keep = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]  # i表示box得分最高的索引
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xmin = np.maximum(x1[i], x1[order[1:]])
        ymin = np.maximum(y1[i], y1[order[1:]])
        xmax = np.minimum(x2[i], x2[order[1:]])
        ymax = np.minimum(y2[i], y2[order[1:]])
        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0, xmax - xmin + 1)
        h = np.maximum(0, ymax - ymin + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        indx = np.where(ovr <= thresh)[0]  # 返回坐标
        order = order[indx + 1]  # 因为ovr数组的长度比order数组少一个（0是选取的基准）,所以这里要将所有下标后移一位
    return keep


dets = np.array([
    [11.5, 12, 311.4, 410.6, 0.85],
    [0.5, 1, 300.4, 400.5, 0.97],
    [200.5, 300, 700.4, 1000.6, 0.65],
    [250.5, 310, 700.4, 1000.6, 0.72]
])
res = nms(dets, 0.5)
print(dets[res])