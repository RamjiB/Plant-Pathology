import keras
import pandas as pd
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser(description='Predict model file')
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

test_df = pd.read_csv('test.csv')
print('Number of train images: ' + str(test_df.shape[0]))

# read all the test images and store it in a variable x
images = test_df['image_id'].values
x = utils.read_images(images)

print(x.shape)

# prediction
model = keras.models.load_model(args.model)
predictions = model.predict(x, verbose=1)
print(predictions.shape)

test_df['healthy'] = predictions[:, 0]
test_df['multiple_diseases'] = predictions[:, 1]
test_df['rust'] = predictions[:, 2]
test_df['scab'] = predictions[:, 3]

print(test_df.shape)

test_df.to_csv('densenet121_kernel_initial_d0.csv', index=False)
