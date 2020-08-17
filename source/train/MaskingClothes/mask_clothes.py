# Import Modules
from module.Mask_RCNN.mrcnn import model as modellib
from module.Mask_RCNN.mrcnn.config import Config
from PIL import Image
import tensorflow as tf
import numpy as np
import warnings
import json
import cv2
import os


# Ignore all warnings
def ignore_warnings():
    # Ignore warnings
    old_v = tf.compat.v1.logging.get_verbosity()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings(action='ignore')


# Resize Image for Mask-RCNN Layer
def resize_image(image_path, image_size):
    temp = cv2.imread(image_path)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return temp


# Refine Masks
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m] is True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois


##############################
# Masking Model (Class)
# Parameter:
# img_size(default: 512)
# threshold(default: 0.7)
# gpu_count(default: 1)
# images_per_gpu(default: 1)
##############################
class Model:
    def __init__(self, img_size=None, threshold=None, gpu_count=None, images_per_gpu=None):
        ignore_warnings()
        # Configuration
        self.MODEL_DIR = "../../../data/weight/mask_rcnn_fashion_0006.h5"
        self.LABEL_DIR = "../../../data/image/mask_rcnn/label_descriptions.json"
        self.MASK_DIR = "../../../module/Mask_RCNN"
        self.NUM_CATS = 46
        if img_size is None:
            self.IMAGE_SIZE = 512
        else:
            self.IMAGE_SIZE = img_size
        with open(self.LABEL_DIR) as f:
            self.label_descriptions = json.load(f)
        self.label_names = [x['name'] for x in self.label_descriptions['categories']]

        # Setup Configuration
        class InferenceConfig(Config):
            NAME = "fashion"
            NUM_CLASSES = self.NUM_CATS + 1  # +1 for the background class
            BACKBONE = 'resnet101'
            IMAGE_MIN_DIM = self.IMAGE_SIZE
            IMAGE_MAX_DIM = self.IMAGE_SIZE
            IMAGE_RESIZE_MODE = 'none'
            RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
            if threshold is None:
                DETECTION_MIN_CONFIDENCE = 0.7
            else:
                DETECTION_MIN_CONFIDENCE = threshold
            if gpu_count is None:
                GPU_COUNT = 1
            else:
                GPU_COUNT = gpu_count
            if images_per_gpu is None:
                IMAGES_PER_GPU = 1
            else:
                IMAGES_PER_GPU = images_per_gpu
        # Execute Inference Configuration
        self.inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference', config=self.inference_config, model_dir=self.MASK_DIR)
        self.model.load_weights(self.MODEL_DIR, by_name=True)

    # Run Method
    def run(self, IMG_DIR):     # IMG_DIR = directory of image (ex: Images/mask1.jpg)
        img = cv2.imread(IMG_DIR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.detect([resize_image(IMG_DIR, self.IMAGE_SIZE)], verbose=1)
        r = result[0]
        if r['masks'].size > 0:
            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'),
                                            (img.shape[1], img.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
            y_scale = img.shape[0] / self.IMAGE_SIZE
            x_scale = img.shape[1] / self.IMAGE_SIZE
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
            masks, rois = refine_masks(masks, rois)
        else:
            masks, rois = r['masks'], r['rois']

        # Declaring return values
        label = []
        score = []
        masked_image = []
        label_type = []

        # Detect whole/upper/lower and Assign complete
        i = 0
        whole = 0
        upper = 0
        lower = 0
        for inx in r['class_ids']:
            category = inx - 1
            temp = img[rois[i][0]:rois[i][2], rois[i][1]:rois[i][3]]
            if category < 5:
                masked_image.append(Image.fromarray(temp))
                label.append(self.label_names[category])
                label_type.append('upper')
                score.append(r['scores'][i])
                upper += 1
            elif category < 9:
                masked_image.append(Image.fromarray(temp))
                label.append(self.label_names[category])
                label_type.append('lower')
                score.append(r['scores'][i])
                lower += 1
            elif category < 13:
                masked_image.append(Image.fromarray(temp))
                label.append(self.label_names[category])
                label_type.append('whole')
                score.append(r['scores'][i])
                whole += 1
            i = i + 1
        if (upper is 1) and (lower is 1):
            complete = True
        elif (whole is 1) and (upper is 1) and (lower is 1):
            complete = True
        elif whole is 1:
            complete = True
        else:
            complete = False

        # Return values
        return img, masked_image, label_type, label, score, complete