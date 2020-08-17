# Import Modules
from imgaug import augmenters as iaa
from module.Mask_RCNN.mrcnn import config as Config
from module.Mask_RCNN.mrcnn import utils, visualize
from pathlib import Path
from sklearn.model_selection import KFold
import module.Mask_RCNN.mrcnn.model as modellib
import sys
import json
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

# Ignore UserWarning - It doesn't work(*)
warnings.simplefilter('ignore', UserWarning)

# Initialize DATA_DIR, ROOT_DIR
DATA_DIR = Path('../../../data/image/mask_rcnn')
ROOT_DIR = Path('../../../module')
sys.path.append(ROOT_DIR/'Mask_RCNN')

# Initialize NUM_CATS, IMAGE_SIZE
NUM_CATS = 46
IMAGE_SIZE = 512

# Import Pretrained Weight
COCO_WEIGHTS_PATH = '../../../data/weight/mask_rcnn_coco.h5'


# Setup Configuration
class FashionConfig(Config.Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1  # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    BACKBONE = 'resnet101'   # resnet50

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    STEPS_PER_EPOCH = 1      # 1000
    VALIDATION_STEPS = 1      # 200


# Execute Configuration
config = FashionConfig()
# config.display()

# Load Label Descriptions to label_descriptions
with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

# From label_descriptions['categories'] to label_names
label_names = [x['name'] for x in label_descriptions['categories']]

# Read train.csv for segmentation
segment_df = pd.read_csv(DATA_DIR/"train.csv")

# Find Multilabel to percent
multilabel_percent = len(segment_df[segment_df['ClassId'].str.contains('_')])/len(segment_df)*100
# print(f"Segments that have attributes: {multilabel_percent:.2f}%")

segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]
# print("Total segments: ", len(segment_df))

# Groupping data in df by Pixels, Height-Width
image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')
# print("Total images: ", len(image_df))


# Resize Image from image_path
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


# Fashion Dataset Class: Create class
class FashionDataset(utils.Dataset):

    def __init__(self, df):
        super().__init__(self)

        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i + 1, name)

        # Add images
        for i, row in df.iterrows():
            self.add_image("fashion",
                           image_id=row.name,
                           path=str(DATA_DIR / 'train' / row.name),
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'],
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x)] for x in info['labels']]

    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height'] * info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            mask[:, :, m] = sub_mask
            labels.append(int(label) + 1)

        return mask, np.array(labels)


# Create dataset for class
dataset = FashionDataset(image_df)
dataset.prepare()

loop = 0
for i in range(loop):
    image_id = random.choice(dataset.image_ids)
    print(dataset.image_reference(image_id))

    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)


# Determine your fold
FOLD = 0
N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df)  # ideally, this should be multilabel stratification


def get_fold():
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]


train_df, valid_df = get_fold()

# DATADIR\train\*
train_dataset = FashionDataset(train_df)
train_dataset.prepare()
valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()

train_segments = np.concatenate(train_df['CategoryId'].values).astype(int)
print("Total train images: ", len(train_df))
print("Total train segments: ", len(train_segments))


plt.figure(figsize=(12, 3))
values, counts = np.unique(train_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, label_names, rotation='vertical')
# plt.show()

valid_segments = np.concatenate(valid_df['CategoryId'].values).astype(int)
print("Total train images: ", len(valid_df))
print("Total validation segments: ", len(valid_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(valid_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, label_names, rotation='vertical')
plt.show()

# Train
# Note that any hyperparameters here, such as LR, may still not be optimal
LR = 1e-4
EPOCHS = [2, 6, 8]

model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5)  # only horizontal flip here
])

############################################
start = time.time()
model.train(train_dataset, valid_dataset,
            learning_rate=LR*2,
            epochs=EPOCHS[0],
            layers='heads',
            augmentation=None)

history = model.keras_model.history.history
end = time.time()
print('Duration:', end-start, 'seconds\n')
############################################

############################################
start = time.time()
model.train(train_dataset, valid_dataset,
            learning_rate=LR,
            epochs=EPOCHS[1],
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]
end = time.time()
print(end-start)
############################################
start = time.time()
model.train(train_dataset, valid_dataset,
            learning_rate=LR/5,
            epochs=EPOCHS[2],
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]
end = time.time()
print(end-start)

############################################
'''
epochs = range(EPOCHS[-1])
plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()
# plt.show()
'''
############################################

best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])