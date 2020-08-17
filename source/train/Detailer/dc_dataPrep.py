import os, glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = "../data/image/Label/"
categories = ["원피스", "블라우스", "코트", "롱자켓", "패딩", "티셔츠", "맨투맨", "니트", "자켓", "가디건",
              "점퍼", "뷔스티", "스웨터", "남방", "스커트", "슬랙스", "린넨팬츠", "데님팬츠"]
num_cat = len(categories)

image_w = 64
image_h = 128

pixels = image_w * image_h * 3

X = []
Y = []

for idx, cat in enumerate(categories):
    # labeling
    label = [0 for i in range(num_cat)]
    label[idx] = 1
    # image directory
    image_dir = DATA_DIR + "/" + cat
    files = glob.glob(image_dir+"/*.gif")
    for i, f in enumerate(files):
        # img -> rgb
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))

        # transform to numpy array
        data = np.asarray(img)
        X.append(data)
        Y.append(label)


X = np.array(X)
Y = np.array(Y)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
xy = (train_X, test_X, train_Y, test_Y)

np.save("./detailer_data.npy", xy)
print("detailer_data.npy has saved!")
