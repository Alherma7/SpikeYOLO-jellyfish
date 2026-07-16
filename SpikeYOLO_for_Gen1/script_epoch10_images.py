"""One-off: render the same val_predictions images the training run would have logged to
TensorBoard, using the train9/weights/best.pt (fitness-best, epoch 8) checkpoint. TB_IMAGE_LOG_
INTERVAL=5 (0-indexed trainer.epoch) only fired at epochs 1 and 6 during the real run, so this
reproduces it post-hoc with identical preprocessing/postprocessing (conf=0.25, iou=0.7, first
unshuffled val batch).
"""
import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from customdataset import CustomDataset, collate_fn
from spike_plotting import render_spike_predictions
from spike_trainer import TIMESTEP, EDGE, IMGSZ, _label_dir
from ultralytics import YOLO
from ultralytics.utils import ops

os.environ['WANDB_DISABLED'] = 'true'

ROOT = Path(__file__).resolve().parent
MODEL_CFG = ROOT / 'snn_yolov8.yaml'
CHECKPOINT = ROOT.parent / 'runs' / 'detect' / 'train9' / 'weights' / 'best.pt'  # best-fitness (epoch 8) weights
VAL_DIR = ROOT / 'jellyFish.v1i.yolov9_prepared' / 'valid' / 'images'
OUT_DIR = ROOT / 'epoch10_val_predictions'
OUT_DIR.mkdir(exist_ok=True)

names = {0: 'C- tuberculata', 1: 'P- noctiluca', 2: 'R- pulmo'}

model = YOLO(str(MODEL_CFG)).load(str(CHECKPOINT))
net = model.model.cuda().eval()

dataset = CustomDataset(img_dir=str(VAL_DIR), label_dir=_label_dir(str(VAL_DIR)),
                         timestep=TIMESTEP, edge=EDGE, imgsz=IMGSZ)
loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

batch = next(iter(loader))
img = batch['img'].cuda().float()

with torch.no_grad():
    raw_preds = net(img)
preds = ops.non_max_suppression(raw_preds, conf_thres=0.25, iou_thres=0.7, multi_label=True, max_det=300)

renders = render_spike_predictions(img.cpu(), batch['cls'], batch['bboxes'], batch['batch_idx'], preds, names)
for i, im in enumerate(renders):
    out_path = OUT_DIR / f'sample_{i}.png'
    cv2.imwrite(str(out_path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    print(f'wrote {out_path}')
