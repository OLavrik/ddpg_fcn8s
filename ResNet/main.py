
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPool2D, Deconv2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute,  LeakyReLU, BatchNormalization, AveragePooling2D, UpSampling2D
from keras.layers import Dense, Flatten, Activation, Dropout, Embedding, Add,Concatenate,MaxPool2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


np.random.seed(42)

a=Cropping2D()

session_config = tf.ConfigProto( )
train_dir = '/home/olgalavricenko/data_set/DuckData/train'

val_dir = '/home/olgalavricenko/data_set/DuckData/val'

test_dir = '/home/olgalavricenko/data_set/DuckData/val'

img_width, img_height = 150, 150


input_shape = (img_width, img_height, 3)

epochs = 30

batch_size = 16

nb_train_samples = 3001

nb_validation_samples = 1372

nb_test_samples = 1372

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
                                              train_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='binary')

val_generator = datagen.flow_from_directory(
                                            val_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

test_generator = datagen.flow_from_directory(
                                             test_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='binary')


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


input_shape = (150, 150, 3)
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
a = Conv2D(4, (1, 1), padding="same", activation="relu")(a)

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
