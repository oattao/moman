import os
import sys
import cv2 as cv 
sys.path.append(os.pardir)

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from configs.image import IMAGE_SIZE, ROTATE_ANGLE, SHIFT_HEIGHT, SHIFT_WIDTH

def square_crop(image_path, resize=True, image_size=IMAGE_SIZE, margin=30,
                return_out=False, croped_image_folder=None):
    """Crop image in image_path, export to the export_folder, with the crop margin
    Args:
        image_path (string): path to the original image file.
        export_folder (string): where to save the croped image
        margin: margin for the object in the image
    """
    # Read the image as gray
    image_path = str(image_path)
    image = cv.imread(image_path, 0)
    
    # Find all contours
    ret, thresh = cv.threshold(image,254,255,cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(thresh, 1, 2)

    # Take the second largest contour
    max_area = max(contours, key = cv.contourArea)
    
    x, y, w, h = cv.boundingRect(max_area)

    # Crop the image
    croped_image = image[y: y+h, x: x+w]

    # Make it square
    max_size = max(w, h)
    desired_size = max_size + 2*margin
    delta_w = desired_size - w
    delta_h = desired_size - h
    top, bottom = delta_h//2, delta_h - delta_h//2
    left, right = delta_w//2, delta_w - delta_w//2
    color = [255, 255, 255]

    croped_image = cv.copyMakeBorder(croped_image, top, bottom, left, right,
                                     cv.BORDER_CONSTANT, value=color)
    if resize:
        croped_image = cv.resize(croped_image, image_size)

    if return_out:
        return croped_image
    else:
        # Save the image
        original_image_name = image_path.split(os.path.sep)[-1]
        croped_image_name = original_image_name

        if not os.path.exists(croped_image_folder):
            os.mkdir(croped_image_folder)
        cv.imwrite(os.path.join(croped_image_folder, croped_image_name), croped_image) 

def prepare_image_for_prediction(image_path ,image_size=IMAGE_SIZE):
    image = Image.open(image_path)
    image = ImageOps.fit(image, image_size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.5) - 1
    image = np.array([image])
    return image

def transform(image_path, transform_type, return_out=False, out_image_folder=None):
    # Read the image as gray
    image_path = str(image_path)
    image = plt.imread(image_path)

    if transform_type == 'rotate':
        angle = np.random.choice(ROTATE_ANGLE, 1, replace=True)[0]
        transformed_image = ndimage.rotate(image, angle, cval=1)
    elif transform_type == 'shift':
        x = np.random.choice(SHIFT_HEIGHT, 1, replace=True)[0]
        y = np.random.choice([1, -1, 2, -2, 3, -3, 4, -4], 1, replace=True)[0]
        z = 0
        transformed_image = ndimage.shift(image, (x, y, z), cval=1)
    elif transform_type == 'flip':
        transformed_image = np.flip(image, axis=1)
    else:
        print('This type of transform is not supported currently.')
        return

    if return_out:
        return transformed_image
    else:
        original_image_name = image_path.split(os.path.sep)[-1]
        transformed_image_name = '{}_{}'.format(transform_type, original_image_name)

        if not os.path.exists(out_image_folder):
            os.mkdir(out_image_folder)
        plt.imsave(os.path.join(out_image_folder, transformed_image_name), 
                    transformed_image) 
        print('Transformed image saved: {}.'.format(transformed_image_name))