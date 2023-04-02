import torch
from collections import Counter
from IoU import intersection_over_union


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
