# Import Modules
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

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

# Save weight file
hdf5_file = "./detailer_model.hdf5"
model.fit(train_X, train_Y, batch_size=32, nb_epoch=10)
model.save_weights(hdf5_file)

# Evaluation of model
score = model.evaluate(test_X, test_Y)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc
