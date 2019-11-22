from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
from urllib.request import urlopen
import numpy as np
import pandas as pd
import os

categories = ["원피스", "블라우스", "코트", "롱자켓", "패딩", "티셔츠", "맨투맨", "니트", "자켓", "가디건",
              "점퍼", "뷔스티", "스웨터", "남방", "스커트", "슬랙스", "린넨팬츠", "데님팬츠"]
num_cat = len(categories)

image_w = 64
image_h = 128

train_X, test_X, train_Y, test_Y = np.load("./model.npy", allow_pickle=True)

train_X = train_X.astype("float") / 256
test_X = test_X.astype("float") / 256


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


pred_look = []

# Read CAMSCON.csv for URL Image
url_df = pd.read_csv('CAMSCON_selected.csv')

for i in range(url_df.__len__()):
    print("\nProcess [%d/%d]" % (i+1, url_df.__len__()))
    url = url_df['url'][i]
    img = Image.open(urlopen(url))
    # Image resize
    img = img.convert("RGB")
    img = img.resize((128, 64))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float") / 256
    X = X.reshape(-1, 128, 64, 3)
    # prediction
    pred = model.predict(X)
    result = [np.argmax(value) for value in pred]  # highest predicted category
    pred_look.append(str(categories[result[0]]))
    print('New data category : ', categories[result[0]])

pred_df = pd.DataFrame(pred_look, columns=['look'])
print(pred_df['look'].value_counts())

