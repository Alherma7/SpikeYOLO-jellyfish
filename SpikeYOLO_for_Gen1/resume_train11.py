import os
from pathlib import Path

from ultralytics import YOLO

from spike_trainer import SpikeDetectionTrainer

os.environ['WANDB_DISABLED'] = 'true'

ROOT = Path(__file__).resolve().parent
LAST = ROOT.parent / 'runs' / 'detect' / 'train11' / 'weights' / 'last.pt'

if __name__ == '__main__':
    model = YOLO(str(LAST))
    print(f"[SpikeYOLO] resuming train11 from: {LAST}")
    model.train(resume=True, trainer=SpikeDetectionTrainer, device=0)
