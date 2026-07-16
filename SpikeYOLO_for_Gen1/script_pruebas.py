import os
from pathlib import Path

from ultralytics import YOLO

from spike_trainer import SpikeDetectionTrainer

os.environ['WANDB_DISABLED'] = 'true'

ROOT = Path(__file__).resolve().parent
MODEL_CFG = ROOT / 'snn_yolov8.yaml'  # explicit path: avoids relying on check_yaml's cwd-dependent
                                      # resolution order, which could otherwise pick the differently
                                      # configured ultralytics/cfg/models/v8/snn_yolov8.yaml instead.
DATA_CFG = ROOT / 'jellyfish_prepared.yaml'  # run prepare_jellyfish_dataset.py first;
                                             # use jellyfish.yaml instead for quick sanity runs
                                             # against the raw, uncurated/unaugmented export
CHECKPOINT = ROOT.parent / 'runs' / 'detect' / 'train10' / 'weights' / 'last.pt'  # continue
                                             # fine-tuning from the previous (interrupted) run's
                                             # latest weights (architecture-matching .load(), not
                                             # resume=True -- train10 was killed mid-epoch 5/11 by
                                             # an unrelated tool call, not a crash; 4 epochs of
                                             # validated progress exist (mAP50-95 0.422 @ epoch4),
                                             # so continue from there instead of from train9 again)

if __name__ == '__main__':
    model = YOLO(str(MODEL_CFG)).load(str(CHECKPOINT))
    print(f"[SpikeYOLO] model yaml: {MODEL_CFG} | ch={model.model.yaml.get('ch')} nc={model.model.yaml.get('nc')}")
    print(f"[SpikeYOLO] continuing from checkpoint: {CHECKPOINT}")

    model.train(
        trainer=SpikeDetectionTrainer,
        data=str(DATA_CFG),
        epochs=11,
        batch=4,  # verified via direct probe: batch=8 needs 10.4GB (exceeds this 8.6GB GPU's VRAM,
                  # triggers slow CPU-RAM spillover); batch=6 peaks at 8.0GB (too little headroom for
                  # EMA/validation on a sustained run); batch=4 peaks at 5.2GB, 1.86s/iter, safe margin.
        imgsz=320,
        device=0,
        workers=4,  # CPU-bound Canny+spikegen encoding in CustomDataset can now bottleneck the
                    # (much faster) GPU forward/backward, so parallelize dataloading across workers.
        optimizer='AdamW',
        lr0=1e-3,
        warmup_epochs=0,
        patience=1000,  # effectively disables early stopping
        save_period=1,  # checkpoint every epoch
        plots=False,  # ultralytics' plot_images() assumes 4D [B,C,H,W] images, not our 5D spike tensors
        close_mosaic=0,  # CustomDataset has no mosaic augmentation to close. Stock BaseTrainer
                          # unconditionally calls self.train_loader.reset() at epoch==(epochs-
                          # close_mosaic) (trainer.py:321) -- a method only ultralytics' own
                          # InfiniteDataLoader wrapper has, not the plain torch DataLoader our
                          # get_dataloader() returns. close_mosaic=0 keeps epoch==(epochs-0) out
                          # of the training loop's range(start_epoch, epochs), so this never fires.
    )
