# https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join, exists
from scipy import stats
from os.path import isfile, join, exists
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm

IMG_SIZE = 200
DIR = 'predict/'


def get_files():
    files = []
    data_files = [f for f in listdir(DIR) if isfile(join(DIR, f))]
    for f in data_files:
        files.append(DIR + f)
    return files


def resize_images():
    for f in get_files():
        im = Image.open(f)
        old_size = im.size
        ratio = float(IMG_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        new_im.paste(im, ((IMG_SIZE - new_size[0]) // 2, (IMG_SIZE - new_size[1]) // 2))
        new_im.save(f, "JPEG")
        print(f + ' resized')


def get_data():
    result = []
    for f in get_files():
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        result.append(im)
    return np.array(result)


def to_grayscale(im, weights=np.c_[0.2989, 0.5870, 0.1140]):
    """
    Transforms a colour image to a greyscale image by
    taking the mean of the RGB values, weighted
    by the matrix weights
    """
    if len(im.shape) != 3:
        return im

    tile = np.tile(weights, reps=(im.shape[0], im.shape[1], 1))
    return np.sum(tile * im, axis=2)


def get_features(X):
    filename = 'features.pickle'

    if exists(filename):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

    resnet_model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)
    resnet_op = AveragePooling2D((7, 7), name='avg_pool_app')(resnet_model.output)
    resnet_model = Model(resnet_model.input, resnet_op, name="ResNet")
    features_array = resnet_model.predict(X)
    features_array = np.reshape(features_array, (-1, features_array.shape[-1]))

    #with open(filename, 'wb') as handle:
    #    pickle.dump(features_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return features_array


resize_images()
x = get_data()
x = get_features(x)
x = np.array([np.ravel(to_grayscale(np.array(i))) for i in x])

with open('svc.model', 'rb') as handle:
    model = pickle.load(handle)


score = model.predict(x)
print(score)
