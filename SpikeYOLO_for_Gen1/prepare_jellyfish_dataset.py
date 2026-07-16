"""One-time preprocessing step for the jellyfish dataset: curation + augmentation.

Mirrors pre_gen1.py's role for the Gen1 pipeline -- builds a new, ready-to-train dataset
directory from the raw source, rather than doing this work at __getitem__ time. Non-destructive
and idempotent: jellyFish.v1i.yolov9/ (the raw Roboflow export) is never modified; re-running
this script always wipes and rebuilds only the OUT directory below.

Steps (see the approved plan for the rationale/decisions behind each):
  1. Copy each of train/valid/test from SRC into OUT.
  2. Remove images matching jellyfish_discard_list.txt from OUT's train and valid splits
     (curation ported from prototipo/preprocess_v2.ipynb + prototipo/spike-main-v2.ipynb).
  3. Augment OUT's train split only: 2x oversampling per class, single-object images only
     (ported from prototipo/augment-utils_v2.ipynb).

Usage: python prepare_jellyfish_dataset.py
"""

import os
import shutil

from dataset_curation import load_discard_list, remove_discarded, count_classes
from augment_utils import augment_dataset

SRC = 'jellyFish.v1i.yolov9'
OUT = 'jellyFish.v1i.yolov9_prepared'
DISCARD_LIST_PATH = 'jellyfish_discard_list.txt'
AUGMENT_MAP = {0: 9, 1: 10, 2: 9}  # class_id -> number of augmented copies per image (0=C-
                                  # tuberculata, 1=P- noctiluca, 2=R- pulmo). For a multi-object
                                  # image, augment_dataset uses the max ratio among the classes
                                  # present -- since P- noctiluca (1) shares many multi-object
                                  # frames with the other two classes, ratios 3/7/3 (original,
                                  # single-object-only proportions from prototipo/spike-main-v2_17_05.ipynb)
                                  # left 0/2 under-augmented relative to 1 once multi-object images
                                  # were included (896/1277/804). Raised here to grow 0/2 rather
                                  # than shrink 1 -- goal is more data overall, not exact parity;
                                  # see actual resulting counts printed by this script. Bumped again
                                  # from 6/7/6 (train9's config) to 9/10/9 -- same +1 relative gap,
                                  # scaled up ~50% -- now that augment_data() draws from 6 transform
                                  # families instead of 3, more copies per image stay less correlated.
NAMES = {0: 'C- tuberculata', 1: 'P- noctiluca', 2: 'R- pulmo'}


def copy_split(split):
    src_img_dir = os.path.join(SRC, split, 'images')
    src_lbl_dir = os.path.join(SRC, split, 'labels')
    out_img_dir = os.path.join(OUT, split, 'images')
    out_lbl_dir = os.path.join(OUT, split, 'labels')

    shutil.rmtree(out_img_dir, ignore_errors=True)
    shutil.rmtree(out_lbl_dir, ignore_errors=True)
    os.makedirs(out_img_dir)
    os.makedirs(out_lbl_dir)

    for fname in os.listdir(src_img_dir):
        if fname.endswith('.jpg'):
            shutil.copy(os.path.join(src_img_dir, fname), os.path.join(out_img_dir, fname))
    for fname in os.listdir(src_lbl_dir):
        if fname.endswith('.txt'):
            shutil.copy(os.path.join(src_lbl_dir, fname), os.path.join(out_lbl_dir, fname))

    return out_img_dir, out_lbl_dir


def main():
    discard_stems = load_discard_list(DISCARD_LIST_PATH)
    print(f'Loaded {len(discard_stems)} discard-list entries from {DISCARD_LIST_PATH}')

    for split in ('train', 'valid', 'test'):
        img_dir, lbl_dir = copy_split(split)
        n_before = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        if split in ('train', 'valid'):
            removed = remove_discarded(discard_stems, img_dir, lbl_dir)
            print(f'{split}: {n_before} -> {n_before - len(removed)} images '
                  f'({len(removed)} discarded)')
        else:
            print(f'{split}: {n_before} images (discard list not applied to test)')

    train_img_dir = os.path.join(OUT, 'train', 'images')
    train_lbl_dir = os.path.join(OUT, 'train', 'labels')

    print('\nPer-class counts before augmentation:')
    count_classes(train_lbl_dir, NAMES)
    augment_dataset(train_img_dir, train_lbl_dir, AUGMENT_MAP)
    print('\nPer-class counts after augmentation:')
    count_classes(train_lbl_dir, NAMES)

    n_final = len([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])
    print(f'\ntrain final image count (curated + augmented): {n_final}')
    print(f'Prepared dataset written to {OUT}/')


if __name__ == '__main__':
    main()
