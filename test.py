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
from urllib.request import urlopen
from PIL import Image
from Mask_RCNN.mrcnn.config import Config
from pathlib import Path
from Mask_RCNN.mrcnn import visualize
import Mask_RCNN.mrcnn.model as modellib
import sys
import json
import random
import cv2
import numpy as np
import pandas as pd
import itertools

# Initialize DATA_DIR, ROOT_DIR
DATA_DIR = Path('')
ROOT_DIR = Path('')
sys.path.append(ROOT_DIR/'Mask_RCNN')

# Initialize NUM_CATS, IMAGE_SIZE
NUM_CATS = 46
IMAGE_SIZE = 512


# Setup Configuration
class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1  # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # DETECTION_NMS_THRESHOLD = 0.0

    STEPS_PER_EPOCH = 1  # 1000
    VALIDATION_STEPS = 1  # 200


# Execute Configuration
config = FashionConfig()
# config.display()


# Load Label Descriptions to label_descriptions
with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

# From label_descriptions['categories'] to label_names
label_names = [x['name'] for x in label_descriptions['categories']]


# Resize Image from image_path
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def resize_url_image(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img

#############################################
# Read url_data to dataFrame choose item you want

# Read CAMSCON.csv for URL Image
url_df = pd.read_csv('url_data/CAMSCON.csv')
url_df.columns = ['url']

# Read FashionGio47.csv for URL Image
# url_df = pd.read_csv('url_data/FashionGio47.csv')
# url_df.columns = ['url', 'title', 'text']

# Read FashionWebzineSnpp(Image).csv for URL Image
# url_df = pd.read_csv('url_data/FashionWebzineSnpp(Image).csv')
# url_df.columns = ['url']

# Read Musinsa.csv for URL Image
# url_df = pd.read_csv('url_data/Musinsa.csv')
# url_df.columns = ['url', 'text']

# Read specific url
# url = 'http://www.fashiongio.com/PEG/15415578699231.jpg'
# url_df = pd.DataFrame({"url": [url]})
#############################################

# Select Weight File manually
model_path = 'fashion20190930T0958/mask_rcnn_fashion_0007.h5'


class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=ROOT_DIR)

# assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Convert data to run-length encoding
def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle


# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois


# URL Image Prediction
Loop = 10   # Set your loop
for i in range(Loop):
    url = random.choice(url_df['url'])
    img = np.array(Image.open(urlopen(url)))
    result = model.detect([resize_url_image(img)], verbose=1)
    r = result[0]
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'),
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        y_scale = img.shape[0] / IMAGE_SIZE
        x_scale = img.shape[1] / IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

        masks, rois = refine_masks(masks, rois)
    else:
        masks, rois = r['masks'], r['rois']

    visualize.display_instances(img, rois, masks, r['class_ids'],
                                ['bg'] + label_names, r['scores'],
                                title=url, figsize=(12, 12))