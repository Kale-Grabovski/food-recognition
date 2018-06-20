# https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
import pickle
import numpy as np
import cv2
from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join, exists
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D
from keras.models import Model

IMG_SIZE = 200
DIR = 'train/'


def get_files():
    files = []
    data_files = [f for f in listdir(DIR) if isfile(join(DIR, f))]
    for f in data_files:
        files.append(DIR + f)
    return files


def resize_images():
    for f in get_files():
        im = Image.open(f)
        w, h = im.size
        if w == IMG_SIZE and h == IMG_SIZE:
            print(f + ' already resized, skipping')
            continue

        old_size = im.size
        ratio = float(IMG_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        new_im.paste(im, ((IMG_SIZE - new_size[0]) // 2, (IMG_SIZE - new_size[1]) // 2))
        new_im.save(f, "JPEG")
        print(f + ' resized')


def get_data():
    filename = 'images.pickle'

    if exists(filename):
        with open(filename, 'rb') as handle:
            result = pickle.load(handle)
            return result['rename'], result['targets']

    result = {
        'rename': [],
        'targets': [],
    }

    for f in get_files():
        print('Processing image', f)

        im = cv2.imread(f, cv2.IMREAD_COLOR)

        result['rename'].append(im)
        result['targets'].append(1 if f.split('/')[-1].split('_')[0] == 1 else -1)

    with open(filename, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return result['rename'], result['targets']


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

    with open(filename, 'wb') as handle:
        pickle.dump(features_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return features_array


resize_images()
x, y = get_data()
x = np.array(x)
y = np.array(y)

features_array = get_features(x)
X_train, X_test, y_train, y_test = train_test_split(features_array, y, random_state=42)

X_train = np.array([np.ravel(to_grayscale(i)) for i in X_train])
X_test = np.array([np.ravel(to_grayscale(i)) for i in X_test])

pca = PCA(svd_solver='randomized', n_components=450, whiten=True, random_state=42)
model = SVC(kernel='rbf', class_weight='balanced')

model.fit(features_array, y_train)
score = model.predict(X_test)
print(classification_report(y_test, score, target_names=['Food', 'Not food']))

with open('svc.model', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
