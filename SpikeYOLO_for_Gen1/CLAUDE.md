# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

SpikeYOLO: a fork of Ultralytics YOLOv8 (`ultralytics/`, v8.0.197) where the backbone/head are
replaced with spiking-neural-network (SNN) modules built on `spikingjelly`. The primary target is
event-camera object detection on the Prophesee **Gen1 Automotive Detection Dataset**, plus an
in-progress experiment applying the same SNN pipeline to a conventional RGB dataset (jellyfish
detection, `jellyFish.v1i.yolov9/`) via spike-rate encoding.

There are effectively three parallel code paths in this repo — know which one a file belongs to
before editing:

1. **Gen1 / event-camera pipeline** — `pre_gen1.py`, `train.py`, `val_gen1.py`, `test.py`,
   `ultralytics/cfg/datasets/gen1.yaml`, `snn_yolov8.yaml`. Uses the standard `ultralytics.YOLO()`
   high-level API (trainer/validator/predictor engine untouched) with the SNN model config.
2. **SNN model implementation** — lives inside the vendored `ultralytics/` package, patched in
   several places to support spiking modules (see Architecture below).
3. **Jellyfish / RGB experiment** — `customdataset.py`, `spike_trainer.py`, `script_pruebas.py`,
   `prepare_jellyfish_dataset.py`. Unlike the Gen1 path, this one *does* go through ultralytics'
   real `Trainer`/`Validator` machinery (`spike_trainer.SpikeDetectionTrainer`/
   `SpikeDetectionValidator`, subclasses of `DetectionTrainer`/`DetectionValidator` that swap in
   `CustomDataset` for the stock `.npy`-based `build_yolo_dataset` pipeline). Uses `snntorch` (not
   `spikingjelly`) for spike encoding of images. `snn_yolov8.yaml`'s `MS_GetT` is intentionally
   disabled here — `CustomDataset`'s dataloader batches are `[B,T,C,H,W]`, and
   `BaseModel._predict_once` (`ultralytics/nn/tasks.py`) centrally transposes to `[T,B,C,H,W]`
   right before the backbone.

## Setup

```bash
conda env create -f yolov8_environment.yaml   # reference env (CUDA 11.6, torch 2.0.0, python 3.9)
pip install -r requirements.txt

cd spikingjelly-0.0.0.0.12
python setup.py install                        # required: spikingjelly is vendored, not on PyPI pinned version
```

The `prophesee-automotive-dataset-toolbox-master.zip` is the Prophesee I/O toolbox
(`prophesee.src.io.psee_loader.PSEELoader`) used by `pre_gen1.py` to read `.dat`/`.npy` event
files — extract it (or ensure `prophesee/` is importable) before running preprocessing.

## Commands

Gen1 pipeline (paths inside these scripts are hardcoded absolute paths from the original authors —
**always check/edit the top of the script** before running, e.g. `path`/`outpath` in `pre_gen1.py`,
the checkpoint path in `test.py`/`val_gen1.py`, and `device=[...]` GPU index lists in all of them):

```bash
python pre_gen1.py     # builds event-frame .npy + YOLO-format .txt labels from raw Gen1 .dat/.npy
python train.py         # model = YOLO("snn_yolov8s.yaml"); model.train(data="gen1.yaml", ...)
python val_gen1.py      # model.val(data="gen1.yaml", ...) on a saved checkpoint
python test.py          # like val, plus registers forward hooks on MultiStepLIFNode layers to
                         # compute per-layer firing rate (printed as fr_dict at the end)
```

No test suite, linter config, or CI is set up in this repo (the vendored `ultralytics/docs` and
`.pre-commit-config.yaml` are upstream artifacts, not actively enforced here).

## Architecture

### Model config → parsed graph

Models are defined as YAML (`snn_yolov8.yaml`, mirrored at
`ultralytics/cfg/models/v8/snn_yolov8.yaml`) using the same backbone/head list-of-layers format as
standard YOLOv8, but referencing SNN-specific module names: `MS_DownSampling`, `MS_ConvBlock`,
`MS_AllConvBlock`, `MS_StandardConv`, `SpikeSPPF`, `SpikeDetect`, `MS_GetT`/`MS_CancelT` (add/remove
the timestep dim `T`). `ch: 1` at the top of the yaml sets a single input channel (event frames /
edge-filtered images), not RGB's 3.

`ultralytics/nn/tasks.py` (`parse_model`, in `DetectionModel`) is the graph builder — it has been
patched to import these SNN module classes and knows how to size them (channel counts, `T`
handling for `MS_GetT`/`MS_CancelT`). Any new SNN layer type must be registered there (import +
elif branch in `parse_model`) as well as in `ultralytics/nn/modules/__init__.py`.

Look for `functional.reset_net(self.model)` calls in `ultralytics/nn/tasks.py` (`DetectionModel`
forward paths, ~lines 78/90/496) — spiking neurons (LIF/PLIF) are **stateful**, so membrane
potentials must be reset between forward passes/batches or state leaks across samples. When
writing any new forward path or standalone script that calls the model directly (see
`script_pruebas.py`), you must reset state yourself.

### Where the actual SNN layers live (naming is misleading)

- `ultralytics/nn/modules/yolo_spikformer.py` — **this is where the SpikeYOLO-specific layers
  actually live**: `MS_ConvBlock`, `MS_AllConvBlock`, `MS_DownSampling`, `MS_StandardConv`,
  `MS_GetT`/`MS_CancelT`, `SpikeConv`, `SpikeSPPF`, `MS_Concat`, `SpikeDetect`, `SpikeDFL`,
  `RepConv`/`SepRepConv`/`SepAllConv`/`SepConv` variants. Despite the name, this file is not the
  Spikformer transformer model — it's the conv-based SpikeYOLO block library.
- `ultralytics/nn/modules/block.py` and `head.py` still contain the **original, un-modified**
  YOLOv8 CNN blocks (`C2f`, `SPPF`, `Detect`, etc.) — these coexist with the SNN versions and are
  used by any non-SNN model yaml.
- `ultralytics/nn/modules/surrogate.py` — surrogate gradient functions for spike
  backpropagation.
- `ultralytics/nn/modules/spikformer_util/` — training utilities lifted from the original
  Spikformer repo (`lr_sched.py`, `lr_decay*.py`, `lars.py`, `pos_embed.py`, `misc.py`,
  `datasets.py`, `crop.py`) — mostly unused by the Gen1 pipeline, kept for reference/reuse.
- `_quan_base_plus.py` (repo root) — quantization base classes (`_Conv2dQ`, `_LinearQ`,
  LSQ-style learnable-step quantization: `alpha`, `round_pass`, `grad_scale`, `truncation`). Used
  by quantized model variants referenced in `snn_yolov8.yaml` scale comments (e.g. "quant_soft").
  Not wired into `parse_model` by default — only relevant when building a quantized variant.

### Gen1 data pipeline specifics (`pre_gen1.py` / `customdataset.py` in `ultralytics/data` terms)

Raw Gen1 data is event streams (`.dat`, read via `PSEELoader`) plus per-timestamp bounding boxes
(`.npy`). `pre_gen1.py`'s `LoadImagesAndLabels.build_dataset` converts each labeled timestamp into
a `T`-step voxel/frame stack: for each of the last 2.5s before a label, it slices `sample_size`
(250,000 µs) into `T` bins, renders each bin as an `[H, W, 3]` frame (127=no event, 255=positive
event, 0=negative event... see `create_data`), stacks to `[T, H, W, 3]`, and writes both the event
tensor (`.npy`) and normalized YOLO-style labels (`.txt`, `class x y w h` in `[0,1]`) to `outpath`.
`gen1.yaml` (`ultralytics/cfg/datasets/gen1.yaml`) then points the standard `ultralytics.YOLO()`
train/val loaders at that `outpath` directory as if it were a normal YOLO image dataset — the `T`
dimension rides along as extra "channels" that `MS_GetT`/`ch:1` handle inside the model.

When editing `pre_gen1.py`, note the hardcoded `sample_size`, `image_shape`, `T`, `path`, `outpath`
constants at the bottom of the file — these must match the `T`/`ch` values expected by the model
yaml you intend to train.

### `close_mosaic` must stay 0 for the jellyfish pipeline

Any `model.train(trainer=SpikeDetectionTrainer, ...)` call needs `close_mosaic=0` explicitly.
Stock `BaseTrainer._do_train` (`ultralytics/engine/trainer.py:321`) unconditionally calls
`self.train_loader.reset()` at `epoch == (epochs - args.close_mosaic)` (default `close_mosaic=10`)
— a method only ultralytics' own `InfiniteDataLoader` wrapper has, not the plain
`torch.utils.data.DataLoader` that `SpikeDetectionTrainer.get_dataloader()` returns. `CustomDataset`
has no mosaic augmentation to close anyway. This only crashes once `epochs > close_mosaic`, so it
won't show up in short smoke tests (e.g. 1-6 epochs) — only in longer real runs.

### TensorBoard for the jellyfish pipeline

`pip install tensorboard` (needed for `torch.utils.tensorboard.SummaryWriter`; not in
`requirements.txt`). Loss/mAP scalar curves are logged automatically by ultralytics' own
`ultralytics/utils/callbacks/tensorboard.py` — no SNN-specific code needed. Validation-image
logging (spike frames with GT/predicted boxes) does *not* come for free: `plot_images()`
(`ultralytics/utils/plotting.py`) assumes 4D `[B,C,H,W]`, and the stock plotting path only fires
on the last training epoch anyway (`BaseValidator.__call__`'s `args.plots &=
trainer.stopper.possible_stop or (epoch == epochs - 1)` gate). `spike_plotting.py` +
`spike_trainer.py`'s `log_spike_val_predictions` (registered on `on_fit_epoch_end`, every
`TB_IMAGE_LOG_INTERVAL` epochs, independent of `args.plots`) fill this gap: collapse each
spike sample's `T` dimension to a single image, draw GT (blue) + predicted (green) boxes, and
push to the same TensorBoard run via `WRITER.add_image`. Launch with
`tensorboard --logdir <trainer.save_dir>`.

### Jellyfish data-prep pipeline (`prepare_jellyfish_dataset.py`)

`jellyFish.v1i.yolov9/` is a raw Roboflow export (3 classes) and is never modified in place.
`prepare_jellyfish_dataset.py` builds `jellyFish.v1i.yolov9_prepared/` from it (safe to re-run —
always wipes and rebuilds just the output dir): copies `train`/`valid`/`test`, then removes any
image listed in `jellyfish_discard_list.txt` (`dataset_curation.remove_discarded`, matched by
exact filename stem — the discard list itself has inconsistent zero-padding between entries, so
do not switch this back to prefix/glob matching) from `train`+`valid` (not `test`), then runs
`augment_utils.augment_dataset` on `train` only (Albumentations affine scale/translate/rotate
applied to image+bboxes together; only single-object images are augmented). `jellyfish.yaml`
points at the raw export (quick sanity runs); `jellyfish_prepared.yaml` points at the prepared
output (`script_pruebas.py`'s default). `spike_utils.py` (noisy-image spike-density filtering) is
ported but intentionally not wired into `prepare_jellyfish_dataset.py` — see that file's docstring.

`prototipo/` holds the original Kaggle-notebook prototype this pipeline (and `dataset_curation.py`/
`augment_utils.py`/`spike_utils.py`/`jellyfish_discard_list.txt`) was ported from. It also contains
a separate, unrelated 15-class prototype (`spike-main-15-spc.ipynb`, `transformacion spike dataset
final.ipynb`) tied to a different Kaggle dataset not present in this repo — don't confuse the two.
Kept for reference/audit only; treat as archived, not active code.

### Firing-rate instrumentation (`test.py`)

`test.py` demonstrates the pattern for inspecting SNN internals: iterate `model.named_modules()`,
tag each `MultiStepLIFNode` with a `.name`, register a `forward_hook` that accumulates
`output.mean() / iter` into a global dict keyed by module name. Reuse this pattern for any new
spike-rate/sparsity analysis rather than modifying the model's forward pass.
