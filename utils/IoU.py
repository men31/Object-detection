import torch


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
    

if __name__ =='__main__':
    pred_bbox = torch.Tensor([[1, 1, 4, 3], [2, 2, 7, 8]])
    label_bbox = torch.Tensor([[2, 2, 7, 8], [1, 1, 4, 3]])
    print(intersection_over_union(pred_bbox, label_bbox, bbox_format='corner'))
