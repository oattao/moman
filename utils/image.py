import numpy as np
from PIL import Image, ImageOps
from configs.image import IMAGE_SIZE2

def load_image(image_path, to_4d, normalized=True, image_size=IMAGE_SIZE2):
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.fit(image, image_size, Image.ANTIALIAS)
    image = np.asarray(image)
    if normalized:
        image = (image.astype(np.float32) / 127.5) - 1
    if to_4d:
        return np.array([image])
    else:
        return image