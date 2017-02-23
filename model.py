import os
import json
import numpy as np
import math
import pandas as pd
import cv2
import h5py
import random
import datetime
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.misc import imsave
from skimage import io
from skimage import transform
from mpl_toolkits.mplot3d import Axes3D

def read_csv(path, random_state=None, reduce_set_size=False):

	data = pd.read_csv(path)
	data = data.values

	if reduce_set_size:
		# Randomly select 75% odata the images with a steering angle odata 0 to remove, in order to avoid overdataitting
		zero_id = np.where(data[:, 3] == 0)
		zero_id = shuffle(zero_id[0], random_state=random_state)
		del_id = zero_id[0:int(0.75*len(zero_id))]
		data = np.delete(data, del_id, axis=0)

	return data

def read_image(f):
	'''Take an array of filenames as input and reads the images with those names.'''

	x = np.array([]).reshape(0, 90, 320, 3)

	if f.ndim == 1:
		for i in f:
			RGB_img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
			RGB_img = RGB_img[50:140, :, :]
			x = np.append(x, np.array([RGB_img]), axis=0)

			# fig = plt.figure()
			# ax = fig.add_subplot(111)
			# ax.imshow(RGB_img)
			# plt.show()
	else:
		for i in range(f.shape[-1]):
			for j in range(len(f)):
				RGB_img = cv2.cvtColor(cv2.imread(f[j, i]), cv2.COLOR_BGR2RGB)
				RGB_img = RGB_img[20:110, :, :]
				x = np.append(x, np.array([RGB_img]), axis=0)

	return x

def rotate_images(data, path, max_angle=20, savefile=False):
	
	count = 1
	camera = ['center', 'left', 'right']
	res = np.array([]).reshape(0, 7)

	for i in data:
		res = np.append(res, np.array([i]), axis=0)

		for j in range(3):
			filename = path + '/' + camera[j] + '_rotated_img_' + str(count) + '.jpg'
			res[-1, j] = filename

			if savefile:
				angle = random.uniform(-max_angle, max_angle)	# Randomly selected angle between max_angle
				img = Image.open(i[j])
				img = img.rotate(angle)
				imsave(filename, img)

		count += 1

	return res		

def shear_images(data, path, max_shear=0.2, savefile=False):

	count = 1
	camera = ['center', 'left', 'right']
	res = np.array([]).reshape(0, 7)

	for i in data:
		res = np.append(res, np.array([i]), axis=0)
		
		for j in range(3):
			filename = path + '/' + camera[j] + '_sheared_img_' + str(count) + '.jpg'
			res[-1, j] = filename

			if savefile:
				# Randomly selected angle between max_angle
				shearing = random.uniform(-max_shear, max_shear)	
				img = io.imread(i[j])
				# Create Afine transform
				afine_tf = transform.AffineTransform(shear=shearing)
				# Apply transform to image data
				img = transform.warp(img, afine_tf)
				imsave(filename, img)
	
		count += 1

	return res		

def shift_brightness_images(data, path, savefile=False):

	count = 1
	camera = ['center', 'left', 'right']
	res = np.array([]).reshape(0, 7)

	for i in data:
		res = np.append(res, np.array([i]), axis=0)

		for j in range(3):
			filename = path + '/' + camera[j] + '_shifted_brightness_img_' + str(count) + '.jpg'
			res[-1, j] = filename

			if savefile:
				random_bright = .25+np.random.uniform()    # Randomly generate a brightness level
				img = cv2.imread(i[j])
				# Apply the random brightness to the V layer of an HSV image and then convert back to RGB
				RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = cv2.cvtColor(RGB_img,cv2.COLOR_RGB2HSV)
				img[:,:,2] = img[:,:,2]*random_bright
				img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
				imsave(filename, img)
	
		count += 1

	return res		

def create_augmented_dataset():
	data = reduce_set_size('driving_log.csv', random_state=0)
	res = rotate_images(data, 'Aug_IMG', savefile=True)
	res = shear_images(data, 'Aug_IMG', savefile=True)
	res = np.append(res, shift_brightness_images(data, 'Aug_IMG', savefile=True), axis=0)

	df = pd.DataFrame(res, columns=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
	df.to_csv('aug_driving_log.csv')

# Create a generator to input data into the model in batches
def generate_arrays_from_file(f, labels, batch_size):
	while True:
		f, labels = shuffle(f, labels)
		for offset in range(0, len(f), batch_size):
			img_file = f[offset:offset+batch_size]
			y = labels[offset:offset+batch_size]
			x = read_image(img_file)
			x, y = shuffle(x, y)

			yield x, y

data = read_csv('driving_log.csv', reduce_set_size=True, random_state=0)
data = np.append(data, read_csv('aug_driving_log.csv', random_state=0), axis=0)
files = data[:, 0]
y = data[:, 3]

# Create training and validation sets
f_train, f_valid, y_train, y_valid = train_test_split(files, y, test_size=0.20, random_state=0)

## Define the architecture
model = Sequential()

# Normalize the data to be within the range (-1, 1)
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(90, 320, 3)))

# Use a modification of the NVIDIA architecture
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
model.add(Activation('relu'))

# model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(1))


# # Use a modification of the commaai network architecture. The original commaai network can be found here:
# # https://github.com/commaai/research/blob/master/train_steering_model.py
# model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
# # model.add(Activation('relu'))
# model.add(ELU())
# model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
# # model.add(Activation('relu'))
# model.add(ELU())
# model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Dropout(.2))
# # model.add(Activation('relu'))
# model.add(ELU())
# model.add(Dense(512))
# model.add(Dropout(.5))
# # model.add(Activation('relu'))
# model.add(ELU())
# model.add(Dense(1))

# Compile and fit the Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit_generator(generate_arrays_from_file(f_train, y_train, 128), 
					samples_per_epoch=len(f_train), 
					nb_epoch=5, verbose=1, validation_data=generate_arrays_from_file(f_valid, y_valid, 128),
					nb_val_samples=len(f_valid))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
