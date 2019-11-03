from pathlib import Path
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import visualize
from flask import Flask, request, render_template
from sklearn.externals import joblib
import Mask_RCNN.mrcnn.model as modellib
import json
import cv2
import imageio
import numpy as np

# Initialize NUM_CATS, IMAGE_SIZE, model_path
NUM_CATS = 46
IMAGE_SIZE = 512
model_path = 'fashion20191028T0500/mask_rcnn_fashion_0001.h5'
ROOT_DIR = Path('/Mask_RCNN')


# Setup Configuration
class InferenceConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1  # +1 for the background class
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = 'resnet101'
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # DETECTION_NMS_THRESHOLD = 0.0
    DETECTION_MIN_CONFIDENCE = 0.65
    STEPS_PER_EPOCH = 1  # 1000
    VALIDATION_STEPS = 1  # 200

inference_config = InferenceConfig()

app = Flask(__name__)
# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


# Load Label Descriptions to label_descriptions
with open("label_descriptions.json") as f:
    label_descriptions = json.load(f)

# From label_descriptions['categories'] to label_names
label_names = [x['name'] for x in label_descriptions['categories']]


# Resize Image from image_path
def resize_image(image_path):
    try:
        img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        print(str(e))


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


def remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list


@app.route('/predict', methods=['POST'])
def handle_request():
    file = request.files['image']
    if not file: return render_template('index.html', label="No Files")

    img = imageio.imread(file)
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    result = model.detect([resize_image(img)], verbose=1)
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
    '''
    visualize.display_instances(img, rois, masks, r['class_ids'],
                                ['bg'] + label_names, r['scores'],
                                title='predict', figsize=(12, 12))
    '''
    r = remove(r['class_ids'])
    list_class = []
    for x in range(r.__len__()):
        list_class.append(label_names[r[x]])
    if not r:
        return render_template('index.html', label='No class')
    else:
        return render_template('index.html', label=list_class)


app.run(host="127.0.0.1", port=8000, debug=True)