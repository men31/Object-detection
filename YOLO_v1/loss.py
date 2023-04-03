import torch
from utils_ori import intersection_over_union


class YoloLoss(torch.nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # There are two iou because there are two bounding boxes
        iou_b1 = intersection_over_union(prediction[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(prediction[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, iou_maxes_idx = torch.max(ious, dim=0)
        print('Val')
        print(iou_maxes_idx)
        exists_box = target[..., 20].unsqueeze(3) # Identity object 1 and 0

        # box coordinates
        box_predictions = exists_box * ((iou_maxes_idx * prediction[..., 26:30] \
                                        + (1-iou_maxes_idx) * prediction[..., 21:25]))
        box_target = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6) * torch.sign(box_predictions[..., 2:4])
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
        )

        # object loss
        pred_box = (iou_maxes_idx * prediction[..., 25:26] + (1 - iou_maxes_idx) * prediction[..., 20:21])
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # no object loss
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * prediction[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        ) + self.mse(
            torch.flatten((1-exists_box) * prediction[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 25:26], start_dim=1)
        )

        # class loss
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * prediction[..., 20], end_dim=-2),
            torch.flatten(exists_box * target[..., 20], end_dim=-2)
        )

        loss = (self.lambda_coord * box_loss + object_loss + 
                self.lambda_noobj * no_object_loss + class_loss)
        return loss