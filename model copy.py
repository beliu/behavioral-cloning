import os
import json
import numpy as np
import math
import pandas as pd
import cv2
import h5py
from PIL import Image
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.layers.core import Reshape
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from mpl_toolkits.mplot3d import Axes3D

def normalize(image_data):
	'''Normalizes the input data to be within the range (-0.5, 0.5)'''
	a = -0.5
	b = 0.5
	min_value = 0
	max_value = 255
	return a + (((image_data - min_value)*(b - a))/(max_value - min_value))

def preprocess_data(path, valid_size=0.2, three_cameras=False):
	'''
		Shuffles and splits the data into a training and validation set. Also
		augments data if the option is enabled.
	'''

	f = pd.read_csv(path)
	f = f.values
	c = 0.2    # Adjustment in steering angle applied to left and right images

	# Randomly select 75% of the images with a steering angle of 0 to remove, in order to avoid overfitting
	zero_id = np.where(f[:, 3] == 0)
	zero_id = shuffle(zero_id[0], random_state=0)
	del_id = zero_id[0:int(0.75*len(zero_id))]
	f = np.delete(f, del_id, axis = 0)

	f_len = len(f)
	if three_cameras:
		f_len = len(f)*3

	# Create an array that contains all the files of each camera angle
	y = np.empty([f_len])
	# Store the center images
	files = f[:, 0]
	y[0:len(f)] = f[:, 3]

	if three_cameras:
		# Store the left images
		files = np.append(files, f[:, 1])
		y[len(f):len(f)*2] = (f[:, 3] + c)
		# Store the right images
		files = np.append(files, f[:, 2])
		y[len(f)*2:] = (f[:, 3] - c)

	# Create training and validation sets
	f_train, f_valid, y_train, y_valid = train_test_split(files, y, test_size=valid_size, 
															random_state=0)

	return f_train, f_valid, y_train, y_valid

def read_image(f):
	'''Take an array of filenames as input and reads the images with those names.'''

	x = np.empty([len(f), 160, 320, 3])
	for i in range(len(f)):
		# x[i] = normalize(cv2.imread(f[i]))
		x[i] = cv2.imread(f[i])

	return x

# Create a generator to input data into the model in batches
def generate_arrays_from_file(f, labels, batch_size):
	while True:
		f, labels = shuffle(f, labels, random_state=0)
		for offset in range(0, len(f), batch_size):
			img_file = f[offset:offset+batch_size]
			y = labels[offset:offset+batch_size]
			x = read_image(img_file)

			yield x, y

# Create a Validation Set for the model
log = 'driving_log.csv'
f_train, f_valid, y_train, y_valid = preprocess_data(log)
valid_set = read_image(f_valid)
samples = len(f_train)
# print(samples)

## Define the architecture
model = Sequential()

# Normalize the data to be within the range (-1, 1)
model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(160, 320, 3),
            output_shape=(160, 320, 3)))


# # Use a modification of the NVIDIA architecture
# model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', input_shape=(160, 320, 3)))
# model.add(ELU())
# model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
# model.add(ELU())
# model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
# model.add(ELU())
# model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
# model.add(ELU())

# model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(ELU())

# model.add(Dense(1164))
# model.add(Dropout(0.5))
# model.add(ELU())

# model.add(Dense(100))
# model.add(Dropout(0.5))
# model.add(ELU())

# model.add(Dense(50))
# model.add(Dropout(0.5))
# model.add(ELU())

# model.add(Dense(10))
# model.add(Dropout(0.5))
# model.add(ELU())

# model.add(Dense(1))


# Use a modification of the commaai network architecture. The original commaai network can be found here:
# https://github.com/commaai/research/blob/master/train_steering_model.py
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(1))

# Compile and fit the Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit_generator(generate_arrays_from_file(f_train, y_train, 128), samples_per_epoch=samples, 
					nb_epoch=5, verbose=1, validation_data = (valid_set, y_valid))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
