import numpy as np


def get_IoU(pred_bboxes, gt_bbox):
    if pred_bboxes.shape[0] > 0:
        xmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
        ymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
        xmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
        ymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])

        w = np.maximum(xmax - xmin + 1, 0)
        h = np.maximum(ymax - ymin + 1, 0)
        inters = w * h
        unions = (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1) + (
                    gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1) - inters
        overlaps = inters / unions
        overlaps_max = np.max(overlaps)
        overlaps_max_idx = np.argmax(overlaps)

        return overlaps, overlaps_max, overlaps_max_idx


if __name__ == "__main__":
    gt_bbox = np.array([70, 80, 120, 150])

    pred_bboxes = np.array([[15, 18, 47, 60],
                            [50, 50, 90, 100],
                            [70, 80, 120, 145],
                            [130, 160, 250, 280],
                            [25.6, 66.1, 113.3, 147.8]])

    overlaps, overlaps_max, overlaps_max_idx = get_IoU(pred_bboxes, gt_bbox)

    print("get_max_IoU:", "overlaps:",overlaps, "\noverlaps_max:", overlaps_max, "\noverlaps_max_idx:", overlaps_max_idx)
