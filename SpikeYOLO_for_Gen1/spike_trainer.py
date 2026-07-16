"""SNN/spike-encoded training via ultralytics' own Trainer/Validator machinery.

Wires `customdataset.CustomDataset` (RGB -> snntorch spike encoding) into
`ultralytics.models.yolo.detect.DetectionTrainer`/`DetectionValidator` instead of the
stock `.npy`-based, Gen1-specific `build_yolo_dataset` pipeline (which assumes different
on-disk data and hardcodes a 320x320 letterbox with augmentation disabled).

Usage: see the bottom of this file, or `spike_train.py`.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from customdataset import CustomDataset, collate_fn
from spike_plotting import render_spike_predictions, log_images
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

# Log a spike-image + GT/predicted-box preview to TensorBoard every N epochs. Independent of
# ultralytics' own `args.plots`, which -- during training-time validation -- only fires on the
# last epoch (BaseValidator.__call__: `self.args.plots &= trainer.stopper.possible_stop or
# (trainer.epoch == trainer.epochs - 1)`), and which calls plot_images() on batch['img'] assuming
# 4D [B,C,H,W] anyway (ours is 5D [B,T,C,H,W]).
TB_IMAGE_LOG_INTERVAL = 5


def log_spike_val_predictions(trainer):
    """on_fit_epoch_end callback: logs the (fixed, unshuffled) first validation batch's
    spike-collapsed images with GT + predicted boxes to TensorBoard, every TB_IMAGE_LOG_INTERVAL
    epochs.
    """
    import ultralytics.utils.callbacks.tensorboard as tb_callback

    if not tb_callback.WRITER or trainer.epoch % TB_IMAGE_LOG_INTERVAL != 0:
        return
    validator = trainer.validator
    if validator is None or validator.dataloader is None:
        return

    batch = next(iter(validator.dataloader))
    batch['img'] = batch['img'].to(trainer.device).float()

    model = trainer.ema.ema or trainer.model
    was_training = model.training
    model.eval()
    with torch.no_grad():
        preds = validator.postprocess(model(batch['img']))
    if was_training:
        model.train()

    renders = render_spike_predictions(batch['img'].cpu(), batch['cls'], batch['bboxes'],
                                        batch['batch_idx'], preds, trainer.data['names'])
    log_images(renders, tag_prefix='val_predictions', step=trainer.epoch)

# Dataset-loading params for CustomDataset. Kept as module constants (not threaded through
# ultralytics' `self.args`) because `get_cfg`/`Model.train(**kwargs)` rejects any keyword that
# isn't one of the predefined DEFAULT_CFG keys.
TIMESTEP = 4
EDGE = True  # must match the model yaml's `ch:` (edge=True -> 1 channel, edge=False -> 3 channels)
IMGSZ = 320


def _label_dir(img_dir):
    """Map a `.../<split>/images` directory to its sibling `.../<split>/labels` directory."""
    p = Path(img_dir)
    return str(p.parent / p.name.replace('images', 'labels'))


class SpikeDetectionValidator(DetectionValidator):
    """DetectionValidator against a CustomDataset spike-encoded split."""

    def build_dataset(self, img_path, mode='val', batch=None):
        return CustomDataset(img_dir=img_path, label_dir=_label_dir(img_path),
                              timestep=TIMESTEP, edge=EDGE, imgsz=IMGSZ)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, mode='val')
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=self.args.workers, collate_fn=collate_fn, pin_memory=True)

    def preprocess(self, batch):
        # Images are already spike-encoded floats in [0,1] (snntorch.spikegen), NOT raw 0-255
        # pixels, so skip the stock `/255` normalization.
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        for k in ('batch_idx', 'cls', 'bboxes'):
            batch[k] = batch[k].to(self.device)
        return batch


class SpikeDetectionTrainer(DetectionTrainer):
    """DetectionTrainer against a CustomDataset spike-encoded split.

    Reuses ultralytics' optimizer/scheduler/EMA/checkpointing/logging and the stock
    `v8DetectionLoss`/mAP metric code unmodified -- only dataset construction, the
    dataloader, and batch normalization are swapped out.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        from ultralytics.utils import DEFAULT_CFG
        super().__init__(cfg or DEFAULT_CFG, overrides, _callbacks)
        self.add_callback('on_fit_epoch_end', log_spike_val_predictions)

    def build_dataset(self, img_path, mode='train', batch=None):
        return CustomDataset(img_dir=img_path, label_dir=_label_dir(img_path),
                              timestep=TIMESTEP, edge=EDGE, imgsz=IMGSZ)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        assert mode in ('train', 'val')
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'),
                           num_workers=self.args.workers, collate_fn=collate_fn, pin_memory=True)

    def preprocess_batch(self, batch):
        # Images are already spike-encoded floats in [0,1] (snntorch.spikegen), NOT raw 0-255
        # pixels, so skip the stock `/255` normalization.
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float()
        return batch

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        from copy import copy
        return SpikeDetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def final_eval(self):
        # Strip optimizer state (keeps checkpoints small) but skip stock BaseTrainer.final_eval's
        # post-training re-validation of best.pt (trainer.py:567, `self.validator(model=f)`).
        # That path reloads the model fresh via AutoBackend, which warms it up with a dummy
        # [B,3,H,W] 4D tensor (validator.py:153) -- our SNN only accepts 5D [T,B,C,H,W], so it
        # crashes (only on GPU: AutoBackend.warmup() skips warmup entirely on CPU, which is why
        # this never showed up in earlier CPU runs). It's also redundant: every epoch's real
        # validation already ran through SpikeDetectionValidator with correct preprocessing, and
        # those metrics are already in results.csv/TensorBoard.
        from ultralytics.utils.torch_utils import strip_optimizer
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)
