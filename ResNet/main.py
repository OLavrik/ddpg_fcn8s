import numpy
from keras.datasets import mnist
from keras.models import Sequential, Input,Model
from keras.layers import Dense, Flatten, Activation, Dropout, LeakyReLU, BatchNormalization, Add, MaxPool2D, Concatenate, AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils



numpy.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

def ResConv(kol_kanal , inp ):
    a = Conv2D(kol_kanal,(1,1) ,padding="same", activation="relu", input_shape=(1,320,320) )(inp)
    b = Conv2D(kol_kanal, (3, 3), padding="same", activation="relu", input_shape=(1, 320, 320))(a)
    c = Conv2D(4*kol_kanal, (1, 1), padding="same", activation=None, input_shape=(1, 320, 320))(b)
    d = Concatenate()([c,inp])
    e=Activation('relu')(d)
    return e

input_shape=(28,28,1)
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
a=Dense(10,activation='softmax')(a)

model = Model(inp0, a)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())
model.fit(X_train, Y_train, batch_size=32, epochs=25, validation_split=0.1, shuffle=True)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

