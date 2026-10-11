import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image, img_size=224):
    """Return RGB pixels in 0-255. Model-specific scaling is inside each model."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def to_features(images):
    return np.ascontiguousarray(images, dtype=np.uint8)

