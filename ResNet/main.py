
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPool2D, Deconv2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute,  LeakyReLU, BatchNormalization, AveragePooling2D
from keras.layers import Dense, Flatten, Activation, Dropout, Embedding, Add,Concatenate,MaxPool2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


np.random.seed(42)


session_config = tf.ConfigProto( )
train_dir = 'cloud_data/duckscapes/train'

val_dir = 'cloud_data/duckscapes/val'

test_dir = 'cloud_data/duckscapes/val'

img_width, img_height = 150, 150


input_shape = (img_width, img_height, 3)

epochs = 30

batch_size = 16

nb_train_samples = 69972

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

def ResConv(kol_kanal , inp ):
    a = Conv2D(kol_kanal,(1,1) ,padding="same", activation="relu", input_shape=(1,320,320) )(inp)
    b = Conv2D(kol_kanal, (3, 3), padding="same", activation="relu", input_shape=(1, 320, 320))(a)
    c = Conv2D(4*kol_kanal, (1, 1), padding="same", activation=None, input_shape=(1, 320, 320))(b)
    d = Concatenate()([c,inp])
    e=Activation('relu')(d)
    return e

input_shape=(150,150,3)
inp0=Input(input_shape)

model = Sequential()


a = MaxPool2D((3,3),2)(inp0)
a = ResConv(32,a)
a = ResConv(32,a)
a = ResConv(32,a)
a = ResConv(64,a)
a = ResConv(64,a)
a = ResConv(64,a)

a=AveragePooling2D((1,1))(a)
a=Flatten()(a)
a=Dense(1000,activation='relu' )(a)

a=Dropout(0.4)(a)
a=Dense(1,activation='softmax')(a)

model = Model(inp0, a)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())


model.fit_generator(
                    train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print( (scores[1]*100))
