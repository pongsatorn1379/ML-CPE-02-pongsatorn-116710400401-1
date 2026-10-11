import hashlib
import numpy as np
from PIL import Image
from preprocessing import preprocess_image

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data(data_path, img_size=224):
    """Like the course example: subfolder names become class labels."""
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    classes = sorted(p.name for p in data_path.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("The dataset needs at least two class folders.")

    images, labels, paths, hashes = [], [], [], []
    seen = {}
    skipped = []
    for label, class_name in enumerate(classes):
        count = 0
        for path in sorted((data_path / class_name).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VALID_EXT:
                continue
            relative = path.relative_to(data_path).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                original_label, original_path = seen[digest]
                if original_label != label:
                    raise ValueError(f"Duplicate image has conflicting labels: {relative}, {original_path}")
                skipped.append({"file": relative, "reason": "duplicate", "original": original_path})
                continue
            try:
                with Image.open(path) as image:
                    pixels = preprocess_image(image, img_size)
            except (OSError, ValueError) as error:
                skipped.append({"file": relative, "reason": str(error)})
                continue
            seen[digest] = (label, relative)
            images.append(pixels)
            labels.append(label)
            paths.append(relative)
            hashes.append(digest)
            count += 1
        print(f"{class_name}: {count} usable images")
        if count < 5:
            raise ValueError(f"Too few usable images in {class_name}; need at least 5.")
    print(f"Skipped files: {len(skipped)}")
    return np.stack(images), np.array(labels), classes, paths, hashes, skipped

