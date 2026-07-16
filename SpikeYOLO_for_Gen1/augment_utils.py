"""RGB-space dataset augmentation, ported from prototipo/augment-utils_v2.ipynb.

Applies a randomly chosen geometric transform (affine or perspective, plus an optional horizontal
flip) to an image and its YOLO-format bounding boxes together, before spike encoding. Unlike the
original prototype, multi-object images are also augmented (see augment_dataset) -- only the
number of copies per image is still driven by augment_map.
"""

from glob import glob
import os

import albumentations as A
import cv2


def extract_data(file_path):
    """Reads a YOLO-format label file into a list of [cls, x, y, w, h] rows."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            parts = [float(e) if '.' in e else int(e) for e in parts]
            data.append(parts)
    return data


def augment_data(image, bboxes):
    """Applies a randomly chosen geometric transform to an image and its bounding boxes together.

    Each call independently samples a horizontal flip and picks one of six transform families:
    plain affine (wide translate range, so the object relocates to different parts of the frame,
    plus scale/rotate), sheared affine (skew-driven distortion), perspective warp (viewpoint
    distortion), elastic deformation, grid distortion, or optical (lens) distortion. This is
    deliberate: augment_dataset calls this function `n` times per source image (n =
    augment_map[class]), and a single fixed narrow transform applied repeatedly produces copies
    that are all the same "shape" of variation and end up looking near-identical -- diversifying
    the transform family/range/flip reduces how correlated those n copies are. All six families
    are purely geometric (pixel-value-preserving, aside from interpolation/fill) -- deliberately
    no brightness/contrast/color jitter here, since the downstream spike-rate encoding is
    sensitive to lighting variation and photometric augmentation risks distorting that signal in
    ways that don't correspond to real-world variation (unvalidated -- see project memory).

    Args:
        image (np.array): image read via OpenCV (BGR).
        bboxes (list): list of boxes, each in YOLO format [x_center, y_center, width, height, class_label].

    Returns:
        tuple: (new_image, new_bboxes) with the same transform applied to both.
    """
    coords = [b[:4] for b in bboxes]
    class_labels = [b[4] for b in bboxes]

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Affine(
                        scale=(0.8, 1.2),
                        translate_percent=(-0.2, 0.2),
                        rotate=(-25, 25),
                        p=1
                    ),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.15, 0.15),
                        rotate=(-10, 10),
                        shear=(-15, 15),
                        p=1
                    ),
                    A.Perspective(scale=(0.03, 0.12), p=1),
                    A.ElasticTransform(alpha=40, sigma=6, p=1),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
                    A.OpticalDistortion(distort_limit=0.3, p=1),
                ],
                p=1
            ),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.0,
            min_area=0.0
        )
    )

    transformed = transform(image=image, bboxes=coords, class_labels=class_labels)

    new_bboxes = []
    for i in range(len(transformed['bboxes'])):
        box = list(transformed['bboxes'][i])
        box.append(transformed['class_labels'][i])
        new_bboxes.append(box)

    return transformed['image'], new_bboxes


def write_bboxes_to_txt(bboxes, filepath):
    """Writes a list of [x, y, w, h, class_label] boxes to a YOLO-format label file."""
    with open(filepath, 'w') as f:
        for x, y, w, h, cls in bboxes:
            f.write(f"{int(cls)} {x} {y} {w} {h}\n")


def augment_dataset(images_path, labels_path, augment_map):
    """Generates augmented copies of every labeled image according to augment_map.

    Scans every label file in labels_path. For an image with a single class present, the
    number of augmented copies is augment_map[class] (as before). For a multi-object image,
    the classes present may have different ratios in augment_map -- since one augmented copy
    duplicates *all* boxes in the image together, the number of copies is the max ratio among
    the classes present, so a minority class sharing a frame with a majority class still gets
    its full boost (the majority class ends up mildly over-augmented in that case, which is
    accepted as a minor side effect rather than tracked per-class across shared images).

    Args:
        images_path (str): directory containing the source images.
        labels_path (str): directory containing the source YOLO labels.
        augment_map (dict): {class_id (int): number_of_augmented_copies}.
    """
    txt_files = glob(os.path.join(labels_path, '*.txt'))

    for txt_file in txt_files:
        txt = extract_data(txt_file)

        if not txt:
            continue

        ratios = [augment_map[cls] for cls, *_ in txt if cls in augment_map]
        if not ratios:
            continue
        n = max(ratios)

        base = os.path.basename(txt_file).replace('.txt', '')
        img_path = os.path.join(images_path, f'{base}.jpg')

        image = cv2.imread(img_path)
        if image is None:
            continue

        bboxes = [[x, y, w, h, cls] for cls, x, y, w, h in txt]

        for i in range(n):
            aug_img, aug_bboxes = augment_data(image, bboxes)

            out_img = os.path.join(images_path, f'{base}_aug_{i}.jpg')
            out_lbl = os.path.join(labels_path, f'{base}_aug_{i}.txt')

            cv2.imwrite(out_img, aug_img)
            write_bboxes_to_txt(aug_bboxes, out_lbl)
