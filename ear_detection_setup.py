import os
import sys
import random
import math
import re
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL.Image import Transpose
from numpy import asarray
from tkinter import filedialog
from tkinter import messagebox
import datetime as dt
from my_CNN_model import load_current_model
from utilities import load_data
from landmarks import put_landmarks

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.ear import ear

print("*******************Initalize Model*******************")
print("*******************Initalize Model*******************")
print("*******************Initalize Model*******************")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "/models/mask_rcnn_ear.h5"  # TODO: update this path

# Configurations
config = ear.BalloonConfig()
EAR_DIR = os.path.join(ROOT_DIR, "datasets/ear")


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

#Notebook Perfernces

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

#Load Validation Dataset

# Load validation dataset
dataset = ear.EarDataset()
dataset.load_ear(EAR_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    
# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
#weights_path = model.find_last()
weights_path = "./models/mask_rcnn_ear_0030.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#Run Detection

'''
image_id = random.choice(dataset.image_ids)
image_id = 7
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
#print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
#                                       dataset.image_reference(image_id)))
print(type([image]))

'''                                       
date = dt.datetime.now()
date_title = str(date.hour) + str(date.minute) + str(date.second)

file_path = filedialog.askopenfilename()

image = Image.open(file_path)
b, g, r = image.split()
image = Image.merge("RGB", (r, g, b))

image = image.transpose(Transpose.ROTATE_270)

numpy_array = asarray(image)

# Run object detection
results = model.detect([numpy_array], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]

dataset.class_names[1] = 'ear'

#x1, y1, x2, y2
y1 = r['rois'][0][0] - 100
x1 = r['rois'][0][1] - 100
y2 = r['rois'][0][2] + 100
x2 = r['rois'][0][3] + 100

if(x1 < 0):
    x1 = 0
if(y1 < 0):
    y1 = 0

cropped = image.crop((x1, y1, x2, y2))
cropped_np = np.array(cropped)

original_path = "./result/" + date_title + "_original.png"
cropped_path = "./result/" + date_title + "_cropped.png"
cropped_resize_path = "./result/" + date_title + "_cropped_resize.png"

cv2.imwrite(original_path, numpy_array)
cv2.imwrite(cropped_path, cropped_np)

#Landmark Detection
model = load_current_model('landmark_detection')

cropped = cropped.resize((224,224))
cropped_224 = np.array(cropped)
cv2.imwrite(cropped_resize_path, cropped_224)

X, Y = load_data(single_img_path=cropped_resize_path)     # please make sure your single image consists of only ear

temp = X[0]
temp = temp[None,:] # adjust the dimensions for the model
prediction = model.predict(temp)

for p in range(len(prediction[0])):     # adjust the landmark points for 224x224 image (they were normalized in range 0 to 1)

    prediction[0][p] = int(prediction[0][p] * 224)

put_landmarks(prediction[0], datatitle=date_title, single_img_path=cropped_resize_path)        # the function to drop landmark points on the image

print("Success!")
