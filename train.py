import pandas as pd
import numpy as np
import cv2
import keras
import os
from progress.bar import Bar

train_df = pd.read_csv('train.csv')
print('Number of train images: ' + str(train_df.shape[0]))

# read all the train images and store it in a variable x
x = []
images = train_df['image_id'].values
bar = Bar('Reading train images', max=len(images))
for image_name in images:
    # read image using cv2 and convert it to RGB
    img = cv2.cvtColor(cv2.imread(os.path.join('images/', image_name+'.jpg')), cv2.COLOR_BGR2RGB)
    # resize the images to size (96,96,3)
    img = cv2.resize(img, (96, 96))
    # append it to x
    x.append(img)
    bar.next()
bar.finish()

# change list to array
x = np.array(x)
print('shape of x: ', x.shape)
