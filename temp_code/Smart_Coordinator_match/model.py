from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

categories = ["대학생룩", "결혼식룩", "힙합룩", "패션테러"]
num_cat = len(categories)

image_w = 64
image_h = 128

train_X, test_X, train_Y, test_Y = np.load("./model.npy", allow_pickle=True)
print(train_X, test_X, train_Y, test_Y)

train_X = train_X.astype("float") / 256
test_X = test_X.astype("float") / 256
print('X_train shape:', train_X.shape)


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

print(model.summary())


hdf5_file = "./model.hdf5"

if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(train_X, train_Y, batch_size=32, nb_epoch=10)
    model.save_weights(hdf5_file)

score = model.evaluate(test_X, test_Y)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc
