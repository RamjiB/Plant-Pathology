import pandas as pd
import numpy as np
import cv2
# import keras
import utils

train_df = pd.read_csv('train.csv')
print('Number of train images: ' + str(train_df.shape[0]))

# read all the train images and store it in a variable x
images = train_df['image_id'].values
x = utils.read_images(images)

# labels for train images
y = np.array(train_df.iloc[:, 1:])
print(y.shape)

# from EDA it was clear that multiple_disease class is really lower than other classes, So augmenting images by
# flipping will help the model to have less biased

train_df['class'] = train_df.iloc[:, 1:].idxmax(axis=1)
aug_image, aug_label = utils.aug_multiple_class(train_df[train_df['class'] == 'multiple_diseases']['image_id'])

print('--------augmented image-------------')
print(aug_image.shape, aug_label.shape)

# appending augmented and real images
x = np.vstack((x, aug_image))
y = np.vstack((y, aug_label))

print(x.shape, y.shape)

del aug_image, aug_label



