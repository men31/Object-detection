import torch
import numpy as np
from collections import Counter


def intersection_over_union(pred_bbox, label_bbox, bbox_format='midpoint'):
    '''
    Calculate intersection over union (iou)

    Parameters:
        pred_bbox (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        label_bbox (tensor): Actual Label of Bounding Boxes (BATCH_SIZE, 4)
        bbox_format (str): midpoint (x, y, w, h) or corner (x1, y1, x2, y2)

    Return:
        (tensor): Intersection over union for all example
    '''
    # Change the midpoint format to corner format
    if bbox_format == 'midpoint':
        box1_x1 = pred_bbox[..., 0:1] - pred_bbox[..., 2:3] /2
        box1_y1 = pred_bbox[..., 1:2] - pred_bbox[..., 3:4] /2
        box1_x2 = pred_bbox[..., 0:1] + pred_bbox[..., 2:3] /2
        box1_y2 = pred_bbox[..., 1:2] + pred_bbox[..., 3:4] /2
        box2_x1 = label_bbox[..., 0:1] - label_bbox[..., 2:3] /2
        box2_y1 = label_bbox[..., 1:2] - label_bbox[..., 3:4] /2
        box2_x2 = label_bbox[..., 0:1] + label_bbox[..., 2:3] /2
        box2_y2 = label_bbox[..., 1:2] + label_bbox[..., 3:4] /2

    elif bbox_format == 'corner':
        box1_x1 = pred_bbox[..., 0:1]
        box1_y1 = pred_bbox[..., 1:2]
        box1_x2 = pred_bbox[..., 2:3]
        box1_y2 = pred_bbox[..., 3:4]
        box2_x1 = label_bbox[..., 0:1]
        box2_y1 = label_bbox[..., 1:2]
        box2_x2 = label_bbox[..., 2:3]
        box2_y2 = label_bbox[..., 3:4]

    intersection_bbox = [torch.max(box1_x1, box2_x1), torch.max(box1_y1, box2_y1), torch.min(box1_x2, box2_x2), torch.min(box1_y2, box2_y2)]
    intersection_area = (intersection_bbox[2] - intersection_bbox[0]).clamp(0) * (intersection_bbox[3] - intersection_bbox[1]).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union_area = box1_area + box2_area - intersection_area + 1
    return intersection_area / union_area

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

def AllPointInterpolatedAP(rec, prec):
    mrec = torch.cat([torch.tensor([0]),rec,torch.tensor([1])])
    mpre = torch.cat([torch.tensor([0]),prec,torch.tensor([0])])
    #Find maximum of precision at each position and it's next position
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    #Find indices where recall value changes
    for i in range(len(mrec) - 1):
        if mrec[i] != mrec[i+1]:
            ii.append(i + 1)
    ap = 0
    #Calculate area at each point recall changes
    for i in ii:
        width  = (mrec[i] - mrec[i - 1]) #Since x axis is recall this can be thought as width
        height = mpre[i] #Since y axis is precision this can be thought as height
        area = width * height
        ap += area
    return ap

def ElevenPointInterpolatedAP(rec, pre):
    recallValues = reversed(torch.linspace(0,1,11,dtype=torch.float64))
    rhoInterp = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all indexs of recall values higher or equal than r
        GreaterRecallsIndices = torch.nonzero((rec >= r),as_tuple=False).flatten()
        pmax = 0
        # If there are recalls above r
        if GreaterRecallsIndices.nelement() != 0:
            #Choose the max precision value, from position min(GreaterRecallsIndices) to the end
            pmax = max(pre[GreaterRecallsIndices.min():])
#         print(r,pmax,GreaterRecallsIndices)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    return ap

def mean_average_precision(pred_bbox, label_bbox, iou_thershold=0.5, bbox_format='corner', num_classes=20):
    '''
    Evaluation the object detections' precision by using mean average precision (mAP)

    Parameters:
        pred_bbox (tensor): Predictions of Bounding Boxes (PREDICTTION_SIZE, 7)
            PREDICTION_SIZE: number of prediction bounding boxes from the model
            7: (id_image (1), class (1), probability of class (1), bounding boxes (4))
        label_bbox (tensor): Ground-Truth of Bounding Boxes (GROUND_TRUTH_SIZE, 7)
            GROUND_TRUTH_SIZE: number of ground truth bounding boxes
            7: (id_image (1), class (1), number 1 (1), bounding boxes (4))
        iou_thershold (int/float): Value for Measurement the Similarity between Bounding Boxes
        bbox_format (str): midpoint (x, y, w, h) or corner (x1, y1, x2, y2)
        num_classes (float) (defualt: 20): number of classes

    Return:
        (float): a mean average precision value
    '''
    average_precision = []
    epsilon = 1e-16
    for c in range(num_classes):
        detections = pred_bbox[pred_bbox[:, 1] == c]
        ground_truths = label_bbox[label_bbox[:, 1] == c]
        amount_gt_bbox = Counter(ground_truths[:, 0].int().tolist())
        for id_img, amount_gt in amount_gt_bbox.items():
            amount_gt_bbox[id_img] = torch.zeros(amount_gt)
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_gt_bbox = len(ground_truths)
        if total_gt_bbox == 0:
            continue
        for detection_idx, a_detection in enumerate(detections):
            # Obtain all ground truth in the detection image
            ground_truths_bboxes = ground_truths[ground_truths[:, 0] == a_detection[:, 0]] 
            num_gt_bboxes = len(ground_truths_bboxes)
            if num_gt_bboxes == 0:
                best_iou = 0
            else:
                ious = intersection_over_union(a_detection[-4:].unsqueeze(0), ground_truths_bboxes[-4:], bbox_format=bbox_format)
                best_iou, best_iou_idx = ious.max(1)
                best_iou, best_iou_idx = best_iou[0], best_iou_idx[0]
            if best_iou > iou_thershold:
                if amount_gt_bbox[a_detection[0].int().item()][best_iou_idx] == 0:
                    TP[detection_idx] = 1
                    amount_gt_bbox[a_detection[0].int().item()][best_iou_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_gt_bbox + epsilon)
        precision = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        average_precision.append(ElevenPointInterpolatedAP(recalls, precision)) 
    return sum(average_precision) / len(average_precision)

