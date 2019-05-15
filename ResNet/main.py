
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPool2D, Deconv2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute,  LeakyReLU, BatchNormalization, AveragePooling2D, UpSampling2D
from keras.layers import Dense, Flatten, Activation, Dropout, Embedding, Add,Concatenate,MaxPool2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from scipy.io import loadmat
np.random.seed(42)
im_width = 150
im_height = 150
border = 5
path_train = '/Users/olgalavricenko/Documents/DuckData/train/'
path_test = './Users/olgalavricenko/Documents/DuckData/val.'

def get_data(path, train=True):
    ids = next(os.walk(path + "image"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
           img = load_img(path + '/image/' + id_, grayscale=True)
    x_img = img_to_array(img)
    x_img = resize(x_img, (150, 150, 1), mode='constant', preserve_range=True)

    if train:
        mask = img_to_array(load_img(path + '/segm/' + id_, grayscale=True))
    mask = resize(mask, (150, 150, 1), mode='constant', preserve_range=True)

    if train:
        return X, y
    else:
        return X
X, y = get_data(path_train, train=True)



def ResConv(kol_kanal, inp):
    a = Conv2D(kol_kanal, (1, 1), padding="same", activation="relu")(inp)
    b = Conv2D(kol_kanal, (3, 3), padding="same", activation="relu")(a)
    c = Conv2D(4 * kol_kanal, (1, 1), padding="same", activation=None)(b)
    d = Concatenate()([c, inp])
    e = Activation('relu')(d)
    return e


def ResDeConv(inp):
    a = UpSampling2D((1, 1))(inp)
    b = UpSampling2D((3, 3))(a)
    c = UpSampling2D((1, 1))(b)

    d = Activation('relu')(c)
    return d


input_shape = (150, 150, 1)
inp0 = Input(input_shape)

model = Sequential()

a = MaxPool2D((3, 3), 2)(inp0)
skip1 = ResConv(32, a)
a = MaxPool2D((3, 3), 2)(skip1)
a = ResConv(32, a)
a = MaxPool2D((2, 2), 2)(a)
a = ResConv(32, a)
a = MaxPool2D((2, 2))(a)

a = ResDeConv(a)
a = Conv2D(256, (1, 1), padding="same", activation="relu")(a)
a = ResDeConv(a)

a = Conv2D(131, (1, 1), padding="same", activation="relu")(a)
a = Cropping2D((3, 3))(a)
a = UpSampling2D((2, 2))(a)
a = Conv2D(1, (1, 1), padding="same", activation="relu")(a)

model = Model(inp0, a)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

model_json = model.to_json()
json_file = open("mnist_model.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("model1.h5")
model.fit_generator(
                    train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print( (scores[1]*100))
