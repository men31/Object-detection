import torch
import numpy as np
from IoU import intersection_over_union


def non_max_suppression(bboxes, iou_thershold, prob_thershold, bbox_format='corner'):
    D = []
    bboxes = [a_bbox.numpy() for a_bbox in bboxes if a_bbox[0] > prob_thershold]
    bboxes = np.array(sorted(bboxes, key=lambda x : x[0], reverse=True))
    r = 1
    while bboxes.shape[0] > 0:
        print('Round:', r)
        print('BBox size:', bboxes.shape)
        max_prob_bbox = bboxes[0]
        bboxes = np.delete(bboxes, 0, axis=0)
        # print(np.ones((bboxes.shape[0], 1)) @ max_prob_bbox[np.newaxis, 1:])
        max_prob_bbox_arr = np.ones((bboxes.shape[0], 1)) @ max_prob_bbox[np.newaxis, 1:]
        # print(max_prob_bbox_arr)
        # print(bboxes[:, 1:])
        print('Result:', bboxes((intersection_over_union(torch.Tensor(max_prob_bbox_arr.copy()), 
                                                         torch.Tensor(bboxes.copy()[:, 1:])) < iou_thershold).numpy()))
        bboxes = bboxes[intersection_over_union(torch.Tensor(max_prob_bbox_arr.copy()), 
                                                         torch.Tensor(bboxes.copy()[:, 1:])) < iou_thershold]

    #     max_prob_bbox = bboxes.pop(0)
    #     left_bboxes = [a_bbox for a_bbox in bboxes 
    #                    if intersection_over_union(torch.Tensor(max_prob_bbox[1:]), torch.Tensor(a_bbox[1:]), 
    #                                               bbox_format=bbox_format) < iou_thershold]
        r += 1
        D.append(max_prob_bbox)
        print('-------------------------------')
    return D
    # return torch.Tensor(bboxes)


if __name__ == '__main__':
    bboxes = torch.Tensor([[0.9, 1, 1, 5, 6], [0.8, 2, 1, 3, 3], [0.4, 4, 4, 8, 6], [0.7, 3, 3, 6, 6]])
    iou_thershold = 0.5
    prob_thershold = 0.5
    print(non_max_suppression(bboxes, iou_thershold, prob_thershold))

