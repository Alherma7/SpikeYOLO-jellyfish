"""Renders spike-encoded validation samples with GT/predicted boxes for TensorBoard.

Standard ultralytics image logging (ultralytics/utils/plotting.py's plot_images) assumes a 4D
[B,C,H,W] RGB image batch, so it can't be reused directly on our 5D [B,T,C,H,W] spike tensors.
The time-collapse + box-drawing technique here is ported from
prototipo/spike-main-v2_17_05.ipynb (cell 61) -- that notebook only ever displayed these via
matplotlib; the TensorBoard logging itself (log_images) is new.
"""

import cv2
import numpy as np


def render_spike_predictions(images, cls, bboxes, batch_idx, preds, names, max_images=4, display_size=640):
    """Renders up to max_images samples as HWC uint8 RGB arrays with GT (blue) + predicted (green) boxes.

    Args:
        images (torch.Tensor): [B,T,C,H,W] spike tensor (CPU).
        cls (torch.Tensor): [N,1] GT class ids for the whole batch.
        bboxes (torch.Tensor): [N,4] GT boxes, normalized xywh, relative to images' own H,W.
        batch_idx (torch.Tensor): [N] which image in the batch each GT row belongs to.
        preds (list[torch.Tensor]): one [Ni,6] tensor per image (x1,y1,x2,y2,conf,cls), already
            in the same H,W pixel space as `images` (no rescale to original image size).
        names (dict): {class_id: class_name}.
        max_images (int): number of samples to render.
        display_size (int): square size to resize each render to, for visibility.

    Returns:
        list[np.ndarray]: HWC uint8 RGB images, one per rendered sample.
    """
    b = images.shape[0]
    n = min(max_images, b)
    _, _, c, h, w = images.shape
    scale_x, scale_y = display_size / w, display_size / h

    renders = []
    for i in range(n):
        accum = images[i].sum(dim=0)  # [C,H,W], sum over T
        accum = accum.mean(dim=0) if c > 1 else accum.squeeze(0)  # -> [H,W]
        accum = accum.numpy()
        img_np = (accum / (accum.max() + 1e-6) * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_color = cv2.resize(img_color, (display_size, display_size))

        gt_mask = (batch_idx == i)
        for cx, cy, bw, bh in bboxes[gt_mask].tolist():
            x1, y1 = (cx - bw / 2) * w * scale_x, (cy - bh / 2) * h * scale_y
            x2, y2 = (cx + bw / 2) * w * scale_x, (cy + bh / 2) * h * scale_y
            cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        for x1, y1, x2, y2, conf, pred_cls in preds[i].tolist():
            x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{names[int(pred_cls)]} {conf:.2f}'
            cv2.putText(img_color, label, (int(x1), max(int(y1) - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        renders.append(img_color)

    return renders


def log_images(images, tag_prefix, step):
    """Pushes rendered images to the active TensorBoard SummaryWriter, if one is running."""
    import ultralytics.utils.callbacks.tensorboard as tb_callback

    if not tb_callback.WRITER:
        return
    for i, img in enumerate(images):
        tb_callback.WRITER.add_image(f'{tag_prefix}/sample_{i}', img, step, dataformats='HWC')
