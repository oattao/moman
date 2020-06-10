import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from configs.image import IMAGE_SIZE2, IMAGE_FILE_EXTENSIONS

def rectangle_image(original_path, original_label, x, y,
					new_size_x, new_size_y, new_path,
					label_left, label_right):
	listfile = os.listdir(os.path.join(*original_path, original_label))
	for filename in listfile:
		img = plt.imread(os.path.join(*original_path, original_label, filename))
		old_size = img.shape[0]

		left_img = np.zeros(shape=(new_size_x, new_size_y, 3), dtype=img.dtype)
		right_img = np.zeros(shape=(new_size_x, new_size_y, 3), dtype=img.dtype)

		x_start = (new_size_x - x) // 2
		y_start = (new_size_y - y) // 2

		left_img[x_start: x_start+x, y_start: y_start+y, :] = img[:x, :y, :]
		right_img[x_start: x_start+x, y_start: y_start+y, :] = np.flip(img[:x, old_size-y:, :], axis=1)

		plt.imsave(os.path.join(*new_path, label_left, 'left_{}'.format(filename)), left_img)
		plt.imsave(os.path.join(*new_path, label_right, 'right_{}'.format(filename)), right_img)

def resplit(original_path, original_label, x, y, new_size, new_path, label_left, label_right):
	listfile = os.listdir(os.path.join(*original_path, original_label))
	for filename in listfile:
		img = plt.imread(os.path.join(*original_path, original_label, filename))
		old_size = img.shape[0]

		left_img = np.zeros(shape=(new_size, new_size, 3), dtype=img.dtype)
		right_img = np.zeros(shape=(new_size, new_size, 3), dtype=img.dtype)

		x_start = (new_size - x) // 2
		y_start = (new_size - y) // 2

		left_img[x_start: x_start+x, y_start: y_start+y, :] = img[:x, :y, :]
		right_img[x_start: x_start+x, y_start: y_start+y, :] = np.flip(img[:x, old_size-y:, :], axis=1)

		plt.imsave(os.path.join(*new_path, label_left, 'left_{}'.format(filename)), left_img)
		plt.imsave(os.path.join(*new_path, label_right, 'right_{}'.format(filename)), right_img)		

def centralize(image_path, x, y, new_size, new_path):
	listfile = os.listdir(os.path.join(*image_path))
	for filename in listfile:
		img = plt.imread(os.path.join(*image_path, filename))
		old_size = img.shape[0]
		x_start = (new_size - x) // 2
		y_start = (new_size - 2*y) // 2
		new_img = np.zeros(shape=(new_size, new_size, 3), dtype=img.dtype)
		new_img[x_start:x_start+x, y_start:y_start+y, :] = img[:x, :y, :]
		new_img[x_start:x_start+x, y_start+y:y_start+2*y, :] = img[:x, old_size-y:, :]
		plt.imsave(os.path.join(*new_path, filename), new_img)


def squeeze(image_path, x, y, new_path):
	listfile = os.listdir(os.path.join(*image_path))
	for filename in listfile:
		img = plt.imread(os.path.join(*image_path, filename))
		old_size = img.shape[0]
		part_1 = img[:x, :y, :]
		part_2 = img[:x, old_size-y:, :]
		new_img = np.concatenate((part_1, part_2), axis=1)
		plt.imsave(os.path.join(*new_path, filename), new_img)
 		
	

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_FILE_EXTENSIONS        