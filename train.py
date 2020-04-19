import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import argparse

parser = argparse.ArgumentParser(description='Inputs for the code')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

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

# split the training data into train and valid
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=22, stratify=y, shuffle=True)

del aug_image, aug_label, x, y

print('--------------training started -----------------')

model = utils.model(train=True)
mcp = ModelCheckpoint('dense121.h5', monitor='val_accuracy', save_best_only= True)
history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_valid, y_valid), callbacks=[mcp])

# plot the train and validation  accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.show()
plt.savefig('accuracy plot')


