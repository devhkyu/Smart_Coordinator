# Import Modules
from module.Mask_RCNN.mrcnn import config as maskconfig
from module.Mask_RCNN.mrcnn import model as maskmodel
from module.Mask_RCNN.mrcnn import visualize
import tensorflow as tf
import numpy as np
import warnings
import json
import cv2
import os

# Ignore warnings
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore')

# Initialize Directories
MODEL_DIR = "../../../data/weight/mask_rcnn_fashion_0006.h5"
LABEL_DIR = "../../../data/image/mask_rcnn/label_descriptions.json"
MASK_DIR = "../../../module/Mask_RCNN"
IMG_DIR = "test1.jpg"

# Initialize NUM_CATS, IMAGE_SIZE
NUM_CATS = 46
IMAGE_SIZE = 512

# Load Label Descriptions to label_descriptions
with open(LABEL_DIR) as f:
    label_descriptions = json.load(f)

# From label_descriptions['categories'] to label_names
label_names = [x['name'] for x in label_descriptions['categories']]


# Setup Configuration
class InferenceConfig(maskconfig):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1  # +1 for the background class
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    BACKBONE = 'resnet101'
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    DETECTION_MIN_CONFIDENCE = 0.70
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Execute Inference Configuration
inference_config = InferenceConfig()

# Load Weight File
model = maskmodel.MaskRCNN(mode='inference', config=inference_config, model_dir=MASK_DIR)
model.load_weights(MODEL_DIR, by_name=True)


# Resize Image from image_path
def resize_image(image_path):
    temp = cv2.imread(image_path)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return temp


# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m] == True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois


# Python code to remove duplicate elements
def remove(duplicate):
    final_list = []
    duplicate_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
        else:
            duplicate_list.append(num)
    return final_list, duplicate_list


# Single Image Masking
img = cv2.imread(IMG_DIR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = model.detect([resize_image(IMG_DIR)], verbose=1)
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
                            title='camera1', figsize=(12, 12))
visualize.display_top_masks(img, masks, r['class_ids'], label_names, limit=8)
