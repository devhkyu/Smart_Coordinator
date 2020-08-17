# Import Modules
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np
import warnings
import cv2
import os

# Ignore warnings
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore')

# Set Image Directory
IMG_DIR = "testImage/test2.jpg"

# Define Image Size
IMAGE_W = 128
IMAGE_H = 64


# Read and Resize Image
def resize_image(image_path):
    temp = cv2.imread(image_path)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    return temp


# Set Categories
categories = ["원피스", "블라우스", "코트", "롱자켓", "패딩", "티셔츠", "맨투맨", "니트", "자켓", "가디건",
              "점퍼", "뷔스티", "스웨터", "남방", "스커트", "슬랙스", "린넨팬츠", "데님팬츠"]
num_cat = len(categories)

# Set Image Size
image_w = 64
image_h = 128

# Set train, test dataSet
train_X, test_X, train_Y, test_Y = np.load("./detailer_data.npy", allow_pickle=True)
train_X = train_X.astype("float") / 256
test_X = test_X.astype("float") / 256

# Construct CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=train_X.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_cat))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Load Weight
hdf5_file = "./detailer_model.hdf5"
model.load_weights(hdf5_file)

# Prediction of Look
pred_category = []

# Image resize
img = resize_image(IMG_DIR)
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 256
X = X.reshape(-1, 128, 64, 3)

# Prediction
pred = model.predict(X)
result = [np.argmax(value) for value in pred]  # highest predicted category
pred_category.append(str(categories[int(result[0])]))
print(categories[int(result[0])])
