"""
Mask R-CNN
- The main Mask R-CNN model implementation.
- Copyright (c) 2017 Matterport, Inc.
- Licensed under the MIT License (see LICENSE for details)
- Written by Waleed Abdulla
- https://github.com/matterport/Mask_RCNN

iMaterialist Fashion 2019 at FGVC6
- https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6
- Fine-grained segmentation task for fashion and apparel

Training Mask R-CNN to be a Fashionista (LB=0.07)
- https://www.kaggle.com/pednoi/training-mask-r-cnn-to-be-a-fashionista-lb-0-07
- This kernel was used

Gachon University Graduation Project
- Choi Hyung-Kyu (Image Processing)
- Jung Yoo-Ji (Image Processing)
- Lee Se-Jin (Text Processing)
"""

# Import Modules
from module.Mask_RCNN.mrcnn import config as maskconfig
from module.Mask_RCNN.mrcnn import model as maskmodel
from module.Mask_RCNN.mrcnn import utils
from sklearn.model_selection import KFold
from imgaug import augmenters as iaa
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import json
import cv2

# Ignore Warnings
warnings.filterwarnings(action='ignore')

# Directories
data_dir = Path('../../../data/image/mask_rcnn')
save_dir = Path('../../../data/weight')

# Initialize NUM_CATS, IMAGE_SIZE
NUM_CATS = 46
IMAGE_SIZE = 512

# Import label_description
with open(data_dir/"label_descriptions.json") as f:
    label_descriptions = json.load(f)
label_names = [x['name'] for x in label_descriptions['categories']]

# Import mask_rcnn_coco
mrCoco = '../../../data/weight/mask_rcnn_coco.h5'

# Import train.csv for segmentation
segment = pd.read_csv(data_dir/"train.csv")
segment['CategoryId'] = segment['ClassId'].str.split('_').str[0]

# Grouping data in dataFrame by Pixels, Height-Width
image = segment.groupby('ImageId')['EncodedPixels', 'CategoryId']\
    .agg(lambda x: list(x)).join(segment.groupby('ImageId')['Height', 'Width'].mean())


# Configuration Class
class FashionConfig(maskconfig.Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    BACKBONE = 'resnet101'
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_NMS_THRESHOLD = 0.8
    STEPS_PER_EPOCH = 1500
    VALIDATION_STEPS = 300


# Fashion Dataset Class
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
                           path=str(data_dir / 'train' / row.name),
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


# Prepare dataset, config
dataset = FashionDataset(image)
dataset.prepare()
config = FashionConfig()

# Functions
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img
def get_fold(fold, splits):
    for i, (train_idx, valid_idx) in enumerate(splits):
        if i == fold:
            return image.iloc[train_idx], image.iloc[valid_idx]


# k-Fold Cross Validation
k = 10
f = 0
kf = KFold(n_splits=k, random_state=42, shuffle=True)
s = kf.split(image)
train, valid = get_fold(f, s)

# Dataset
train_dataset = FashionDataset(train)
train_dataset.prepare()
valid_dataset = FashionDataset(valid)
valid_dataset.prepare()

# Load Model
model = maskmodel.MaskRCNN(mode='training', config=config, model_dir=save_dir)
model.load_weights(mrCoco, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
augmentation = iaa.Sequential([iaa.Fliplr(0.5)])

# Training Parameters
LR = 1e-4
EPOCHS = [2, 6, 8]

# Training
model.train(train_dataset, valid_dataset,
            learning_rate=LR*2, epochs=EPOCHS[0],
            layers='heads', augmentation=None)
history = model.keras_model.history.history

model.train(train_dataset, valid_dataset,
            learning_rate=LR, epochs=EPOCHS[1],
            layers='all', augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history:
    history[k] = history[k] + new_history[k]

model.train(train_dataset, valid_dataset,
            learning_rate=LR/5, epochs=EPOCHS[2],
            layers='all', augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history:
    history[k] = history[k] + new_history[k]

best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])
