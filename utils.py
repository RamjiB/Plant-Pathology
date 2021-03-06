import numpy as np
import cv2
import os
from progress.bar import Bar
import keras


def read_images(images):
    x = []
    bar = Bar('Reading train images', max=len(images))
    for image_name in images:
        # read image using cv2 and convert it to RGB
        img = cv2.cvtColor(cv2.imread(os.path.join('images/', image_name + '.jpg')), cv2.COLOR_BGR2RGB)
        # resize the images to size (96,96,3)
        img = cv2.resize(img, (96, 96))
        # append it to x
        x.append(img)
        bar.next()
    bar.finish()
    # change list to array
    x = np.array(x)
    print('shape of x: ', x.shape)
    return x


def aug_multiple_class(images):

    x = []
    y = []
    bar = Bar('Augmentation', max=len(images))
    for image in images:
        # read image using cv2 and convert it to RGB
        img = cv2.cvtColor(cv2.imread(os.path.join('images/', image + '.jpg')), cv2.COLOR_BGR2RGB)
        # resize the images to size (96,96,3)
        img = cv2.resize(img, (96, 96))
        # flip along x axis
        x.append(cv2.flip(img, 0))
        y.append(np.array((0, 1, 0, 0)))
        # flip along y axis
        x.append(cv2.flip(img, 1))
        y.append(np.array((0, 1, 0, 0)))
        # flip along both axis
        x.append(cv2.flip(img, -1))
        y.append(np.array((0, 1, 0, 0)))

        bar.next()
    bar.finish()
    x = np.array(x)
    y = np.array(y)
    return x, y


def model(train=False):
    bm = keras.applications.densenet.DenseNet121(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

    for layer in bm.layers:
        layer.trainable = train

    x = bm.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = keras.layers.Dense(4, activation='softmax')(x)

    m = keras.models.Model(inputs=bm.input, outputs=x)
    m.compile(keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()

    return m
