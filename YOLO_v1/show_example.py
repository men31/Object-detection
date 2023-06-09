import torch
import torchvision.transforms as transform
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils_ori import *
from loss import YoloLoss


seed = 463
torch.manual_seed(seed)

#Hyperparameter
LEANING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('on device:', DEVICE)
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = 'overfit.pth.tar'
IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'

class Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, bboxes):
        for t in self.transform:
            img, bboxes = t(img), bboxes
        return img, bboxes
    
transform = Compose([transform.Resize((448, 448)), transform.ToTensor(), ])

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEANING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    
    train_dataset = VOCDataset(
        "data/100examples.csv", # can be modified
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", # can be modified  
        transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    state_dict = torch.load(LOAD_MODEL_FILE)
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer'])

    for x, y in test_loader:
           x = x.to(DEVICE)
           
           for idx in range(2):
            #    print(y[idx].shape)
               bboxes = cellboxes_to_boxes(model(x))
               bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
               plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

           import sys
           sys.exit()

        
    # for epoch in range(EPOCHS):
    #     # for x, y in train_loader:
    #     #    x = x.to(DEVICE)
    #     #    for idx in range(8):
    #     #        bboxes = cellboxes_to_boxes(model(x))
    #     #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    #     #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    #     #    import sys
    #     #    sys.exit()


    #     pred_boxes, target_boxes = get_bboxes(
    #         train_loader, model, iou_threshold=0.5, threshold=0.4
    #     )

    #     mean_avg_prec = mean_average_precision(
    #         pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    #     )
    #     print(f"Train mAP: {mean_avg_prec}")

    #     if mean_avg_prec > 0.9:
    #        checkpoint = {
    #            "state_dict": model.state_dict(),
    #            "optimizer": optimizer.state_dict(),
    #        }
    #        save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
    #        import time
    #        time.sleep(10)

if __name__ == "__main__":
    main()