import torch
import numpy as np
from IoU import intersection_over_union


def non_max_suppression(bboxes, iou_thershold, prob_thershold, bbox_format='corner'):
    '''
    Cleaning the bounding box by non-max-suppresion (nms) for 1 class only

    Parameters:
        bboxes (tensor): Bounding Boxes with Probability (BATCH_SIZE, 5)
            BATCH_SIZE: number of bounding boxes with same object class
            5: (probability of object class (1), bounding boxes (4))
        iou_thershold (int/float): Value for Measurement the Similarity between Bounding Boxes
        prob_thershold (float [0, 1]): Probabilty Thershold
        bbox_format (str): midpoint (x, y, w, h) or corner (x1, y1, x2, y2)

    Return:
        D (list[tensor]): List of Cleaning Bounding Boxes
    '''
    D = []
    bboxes = [a_bbox.numpy() for a_bbox in bboxes if a_bbox[0] > prob_thershold]
    bboxes = torch.from_numpy(np.array(sorted(bboxes, key=lambda x : x[0], reverse=True)))
    while bboxes.shape[0] > 0:
        max_prob_bbox = bboxes[0]
        dummy_bboxes = bboxes[1:]
        bboxes = dummy_bboxes
        max_prob_bbox_arr = torch.mul(torch.ones((bboxes.shape[0], 1)), max_prob_bbox)
        nms_idx = intersection_over_union(max_prob_bbox_arr[..., 1:], bboxes[..., 1:], bbox_format=bbox_format) < iou_thershold
        bboxes = bboxes[nms_idx.squeeze()].squeeze(axis=1)
        D.append(max_prob_bbox)
    return D


if __name__ == '__main__':
    bboxes = torch.Tensor([[0.9, 1, 1, 5, 6], [0.8, 2, 1, 3, 3], [0.4, 4, 4, 8, 6], [0.7, 3, 3, 6, 6]])
    iou_thershold = 0.5
    prob_thershold = 0.5
    print(non_max_suppression(bboxes, iou_thershold, prob_thershold))

