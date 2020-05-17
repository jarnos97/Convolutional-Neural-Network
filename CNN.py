from PIL import Image
import os
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import h5py
import pandas as pd
import timeit
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import logging
logging.getLogger('tensorflow').disabled = True


#%%
"""
Creating y 
In the original paper, all data is used. However, the authors have not released the y labels of the test set.
Therefore, a new split will be made based on the data that is available (3609 images). 
"""


def retrieve_y(json_folder):
    n_humans = []
    for filename in os.listdir(json_folder):
        path = json_folder + '/' + filename
        with open(path) as f:
            data = json.load(f)
            n_humans.append(data['human_num'])
    n_humans = np.array(n_humans)
    n_humans = n_humans.astype('float32')
    return n_humans


y = retrieve_y('C:/Users/jarno/Documents/MDSE/Semester 2/Deep Learning/Project/NWPU-Crowd/jsons')
# Save y to disk
pickle.dump(y, open('y.pickle', 'wb'))

# Inspecting the distribution of our y
print(f"Range of y: ({min(y)}, {max(y)})")
y_temp = [i for i in y if i < 2000]
fig, axx = plt.subplots(1, 2, figsize=(15, 5))
sns.distplot(y, ax=axx[0])
axx[0].set_title('y')
sns.distplot(y_temp, ax=axx[1])
axx[1].set_title('y')
fig.show()

# How many negative samples are there?
negative_samples = [i for i in y if i == 0]
print(len(negative_samples), len(negative_samples)/3609*100, '%')

# scale y? Could use the MinMAxScaler to scale, could provide better results.
# However, would also mean the MAE becomes much less informative


#%%
""" 
Create x
Resizing images-
Choices made: I chose to make the images of size 128X128 in order to save memory. In addition, the images are made 
square, which makes them lose their aspect ratio. I chose this over padding the pictures since this would require 
the network to learn that the padding is not relevant for the prediction, which in turn would take more epochs.
Each image is converted to an array of shape (128,128,3) and added to the list of converted images.
"""
# Let's look at an example picture
img = Image.open('C:/Users/jarno/Documents/MDSE/Semester 2/Deep Learning/Project/NWPU-Crowd/images/0001.jpg')
print(img.size)
img.show()
img.close()


# Now, we will convert the images to arrays and resize them
def resizing_images(image_folder, max_size):
    """
    The function loops through each file in the picture folder (it assumes only pictures are present). The pictures are
    resized to max_size X max_size pixels. If the image has a RGBA mode, it is converted to RGD in order to be able to
    save it as jpg.
    :param image_folder:
    :param max_size:
    :return:
    """
    images = []
    for filename in os.listdir(image_folder):
        path = image_folder + '/' + filename
        image = Image.open(path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        new_image = image.resize((max_size, max_size))
        data = np.asarray(new_image)
        images.append(data)
        image.close()
    images = np.asarray(images)
    images = images.astype('float32') / 255
    return images


X = resizing_images('C:/Users/jarno/Documents/MDSE/Semester 2/Deep Learning/Project/NWPU-Crowd/images',  256)
# Save X to disk
pickle.dump(X, open('X.pickle', 'wb'))


#%%
"""
Splitting the data
We will maintain (roughly) the same split sizes (%) as the NWPU paper. Meaning a training set of 60%, validation set 
of 10%, and test set of 30%. Like in the paper, the images will simply be split by placing the first 60% of the images
in the training set. The sampling is not done randomly, since there is no order to the images.
"""
X_train = X[:1865]
X_val = X[1865:2176]
X_test = X[2176:3109]
y_train = y[:1865]
y_val = y[1865:2176]
y_test = y[2176:3109]

#%%
"""
Experimenting with the optimizer and number of convolution layers
Performance metrics: MAE, MSE
"""


def convnet_size(optim, loss_func, n_conv_layers):
    """
    The function builds a CNN. The optimizer, loss function and number of convolution layers are specified with the
    parameters. The minimal number of convolution layers is 1, each additional convolution layers is accompanied by a
    MaxPooling layer.
    :param optim:
    :param loss_func:
    :param n_conv_layers:
    :return:
    """
    model = Sequential()
    # Convolution and Maxpooling layers
    model.add(layers.Input((256, 256, 3)))
    # Experimenting with the amount of convolution layers
    for i in range(n_conv_layers-1):
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # output shape after layer: (12,12,32) (size 4608)
    # Flatten output
    model.add(layers.Flatten())
    # Add dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optim, loss=loss_func, metrics=['MeanSquaredError'])
    return model


# We will train models (3 epochs) for two optimizers and 5 sizes of the model (amount of conv layers and maxpooling)
param_dict = {'optim': ['adam', 'rmsprop'], 'n_conv_layers': [2, 3, 4, 5, 6]}
result_dict = {}
for optimizer in param_dict['optim']:
    for n_conv_layer in param_dict['n_conv_layers']:
        cnn = convnet_size(optim=optimizer, loss_func='mean_absolute_error', n_conv_layers=n_conv_layer)
        cnn.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_val, y_val))
        result_dict[f'{optimizer}_{n_conv_layer}'] = [cnn.history.history['loss'][2], cnn.history.history['val_loss'][2]]

print(result_dict.items())
# Overall, the n_conv_layer does not seem to have a very large effect on MAE. We will use 5 convolution layers. This
# seems to be a good trade of between complexity of the model and and sensitivity to changes in the feature map.
# Furthermore, this greatly decreases the amount of trainable parameters compared to less convolution layers.
# The adam optimizer performs better in every case, thus we will use for the other models.


#%%
"""
Grid Search: Epochs and batch size
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
"""


def create_model():
    model = Sequential()
    # Convolution and Maxpooling layers
    model.add(layers.Input((256, 256, 3)))
    # Convolution layer gets 32 filters of size (3x3) (filter size should be ann odd number)
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # Maxpooling layer get size (2, 2), with stride = pool size and padding = valid, meaning no zero padding is applied
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))  # down samples the feature map
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # output shape after layer: (28,28,32) (size 25088)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # output shape after layer: (12,12,32) (size 4608)
    # Flatten output
    model.add(layers.Flatten())
    # Add dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    # The last layer is size 1, since the output is a continuous value. Also, we do not specify an activation function
    # since it is a regression task and the y-values are not transformed
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


cnn = KerasClassifier(build_fn=create_model, verbose=0)
param_grid = {'batch_size': [32, 64, 96], 'epochs': [10, 50, 100]}
grid = GridSearchCV(estimator=cnn, param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=3)
grid_result = grid.fit(X_val, y_val)
print(grid_result)


#%%
"""
Grid Search Learning Rate &  momentum
"""
cnn = KerasClassifier(bui)

#%%
"""
Grid Search Dropout
"""


#%%
"""
Train Initial Model
"""


#%%
"""
Random OverSampling
"""


#%%
"""
Different loss function 1
"""


#%%
"""
Different loss function 2
"""


#%%
def convnet(optim, loss_func):
    model = Sequential()
    # Convolution and Maxpooling layers
    model.add(layers.Input((256, 256, 3)))
    # Convolution layer gets 32 filters of size (3x3) (filter size should be ann odd number)
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # Maxpooling layer get size (2, 2), with stride = pool size and padding = valid, meaning no zero padding is applied
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))  # down samples the feature map
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # output shape after layer: (28,28,32) (size 25088)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # output shape after layer: (12,12,32) (size 4608)
    # Flatten output
    model.add(layers.Flatten())
    # Add dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    # The last layer is size 1, since the output is a continuous value. Also, we do not specify an activation function
    # since it is a regression task and the y-values are not transformed
    model.add(layers.Dense(1))
    model.compile(optimizer=optim, loss=loss_func)
    return model

# Initialize model
cnn = convnet(optim='adam', loss_func='mean_absolute_error')
start = timeit.timeit()
# explain epochs and batch size
cnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))
end = timeit.timeit()
print("Fitting time:", end-start)

# Saving the model and weights to disk
cnn_json = cnn.to_json()
with open("cnn_2.json", "w") as json_file:
    json_file.write(cnn_json)
cnn.save_weights("cnn_2.h5")



#%%
"""
Making Predictions on the first model
"""
# Predicting y on test-set
preds = cnn.predict(X_test)
preds = np.reshape(preds, (933,))
cnn_df = pd.DataFrame({"y_test": y_test, "predictions": preds})

# Add absolute difference
cnn_df['abs_difference'] = abs(y_test - preds)

# MAE
print("MAE:", cnn_df['abs_difference'].mean())  # still quite large

# Negative samples
negatives_df = cnn_df.loc[cnn_df['y_test'] == 0]
print("MAE negative samples:", negatives_df['abs_difference'].mean())
positives_df = cnn_df.loc[cnn_df['y_test'] > 0]
print("MAE positive samples:", positives_df['abs_difference'].mean())

# add dropout ; it is usually set between 0.2 and 0.5.
# tweak learning rate
# momentum (not at the same time with learning rate
# convert y - using minmaxscaler

# if the y is transformed use linear activation function in the last layer. Otherwise, (for the actual y values)
# do not specify a activation function

print(cnn.history['loss'])