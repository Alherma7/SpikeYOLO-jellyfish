"""Spike-encoding / noisy-image quality-filter utilities, ported from prototipo/spike-utils_v2.ipynb.

Available tooling, not currently wired into prepare_jellyfish_dataset.py -- the automatic
detect_noisy() heuristic was tuned by eye on a different encoding run and could flag a
different, unverified set of images on the current dataset. Use these functions manually
(e.g. generate_spikes + detect_noisy + show_noisy_images to review candidates) before deciding
whether to fold any of them into the automatic pipeline.
"""

import math
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from snntorch import spikegen

from customdataset import canny

resize = transforms.Resize((320, 320))
to_tensor = transforms.ToTensor()


def spike_from_image(img_path, timestep=4):
    """Reads an image, Canny-edges it, and rate-codes it to spikes -- returns a [T,C,H,W] tensor."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = canny(img)

    img = Image.fromarray(edges)
    img = resize(img)
    img = to_tensor(img)
    return spikegen.rate(img, num_steps=timestep)


def generate_spikes(img_dir, save_dir):
    """Spike-encodes every .jpg in img_dir, saves a collapsed-over-time PNG preview to save_dir.

    Returns a dict {filename: spike_tensor}.
    """
    os.makedirs(save_dir, exist_ok=True)
    spike_trains = {}

    for f in os.listdir(img_dir):
        if f.endswith('.jpg'):
            spikes = spike_from_image(os.path.join(img_dir, f))
            spike_img = (spikes.sum(dim=0)[0] > 0).numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(save_dir, f.replace('.jpg', '_spike.png')), spike_img)
            spike_trains[f] = spikes

    return spike_trains


def detect_noisy(spike_trains, threshold=0.20):
    """Flags filenames whose spike density (fraction of active pixel-timesteps) >= threshold."""
    return [
        fname
        for fname, spike_map in spike_trains.items()
        if (spike_map.sum() / spike_map.numel()) >= threshold
    ]


def move_noisy_spikes(noisy_list, pre_dir, noisy_dir):
    """Moves the *_spike.png preview for each flagged filename from pre_dir into noisy_dir."""
    os.makedirs(noisy_dir, exist_ok=True)

    for fname in noisy_list:
        spike = fname.replace('.jpg', '_spike.png')
        shutil.move(os.path.join(pre_dir, spike), os.path.join(noisy_dir, spike))


def build_clean_dataset(img_dir, lbl_dir, clean_img_dir, clean_lbl_dir, noisy_list):
    """Rebuilds clean_img_dir/clean_lbl_dir with every file from img_dir/lbl_dir except noisy_list."""
    noisy_set = set(noisy_list)

    shutil.rmtree(clean_img_dir, ignore_errors=True)
    shutil.rmtree(clean_lbl_dir, ignore_errors=True)
    os.makedirs(clean_img_dir)
    os.makedirs(clean_lbl_dir)

    for fname in os.listdir(img_dir):
        if fname not in noisy_set:
            shutil.copy(os.path.join(img_dir, fname), os.path.join(clean_img_dir, fname))

    for fname in os.listdir(lbl_dir):
        if fname.replace('.txt', '.jpg') not in noisy_set:
            shutil.copy(os.path.join(lbl_dir, fname), os.path.join(clean_lbl_dir, fname))


def show_noisy_images(noisy_list, noisy_dir, n=50, cols=3):
    """Displays up to n spike-encoded previews flagged as noisy, for manual review."""
    subset = noisy_list[:n]
    rows = math.ceil(len(subset) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axes = axes.flatten()

    for ax, fname in zip(axes, subset):
        spike_img = cv2.imread(os.path.join(noisy_dir, fname.replace('.jpg', '_spike.png')),
                                cv2.IMREAD_GRAYSCALE)
        ax.imshow(spike_img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_clean_and_spike(clean_dir, spike_dir, n=10):
    """Displays clean-image / spike-preview pairs side by side, for manual review."""
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.jpg')][:n]

    for fname in clean_files:
        clean_img = cv2.cvtColor(cv2.imread(os.path.join(clean_dir, fname)), cv2.COLOR_BGR2RGB)
        spike_img = cv2.imread(os.path.join(spike_dir, fname.replace('.jpg', '_spike.png')),
                                cv2.IMREAD_GRAYSCALE)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(clean_img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(spike_img, cmap='gray')
        plt.axis('off')
        plt.show()
