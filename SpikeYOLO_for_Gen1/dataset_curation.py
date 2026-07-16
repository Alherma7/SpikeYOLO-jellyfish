"""Dataset curation utilities, ported from prototipo/preprocess_v2.ipynb.

Ported from a prototype that matched files via `glob(base_name + "*")` -- a naive prefix
glob. The discard list has inconsistent zero-padding between entries (e.g. "..._00007" vs
"..._000015"), so a prefix match risks sweeping up unrelated, longer-numbered filenames that
merely start with the same digits (e.g. "..._00003*" also matching "..._000030"). This version
matches on the exact pre-Roboflow-suffix stem instead.
"""

import os


def load_discard_list(path):
    """Reads a discard list file (one filename stem per line, '#' comments/blank lines ignored)."""
    stems = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                stems.add(line)
    return stems


def image_stem(filename):
    """Strips the Roboflow export suffix ('_jpg.rf.<hash>.jpg') to recover the original stem."""
    return filename.split('_jpg.rf.')[0]


def remove_discarded(discard_stems, img_dir, label_dir):
    """Removes every image/label pair in img_dir/label_dir whose stem is in discard_stems.

    Matches on the exact stem (see module docstring) rather than a prefix glob, so entries
    like "pelagia04_IMG_00007" cannot accidentally also match "pelagia04_IMG_000070".

    Returns the sorted list of stems that were actually found and removed.
    """
    removed = []
    for filename in os.listdir(img_dir):
        if not filename.endswith('.jpg'):
            continue
        stem = image_stem(filename)
        if stem in discard_stems:
            os.remove(os.path.join(img_dir, filename))
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                os.remove(label_path)
            removed.append(stem)
    return sorted(removed)


def count_classes(label_dir, names):
    """Tallies per-class label frequency across every .txt file in label_dir.

    Args:
        label_dir (str): directory containing YOLO-format .txt labels.
        names (dict): {class_id (int): class_name (str)}.

    Returns:
        dict: {class_id (int): count}.
    """
    counts = {}
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, filename), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                counts[cls_id] = counts.get(cls_id, 0) + 1

    for cls_id, count in sorted(counts.items()):
        print(f'{cls_id} - {names[cls_id]}: {count}')

    return counts
