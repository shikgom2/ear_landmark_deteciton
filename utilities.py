import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def load_data(single_img_path='give a path please'):

    img_path = single_img_path

    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    X = x
    Y = 0

    return X, Y
