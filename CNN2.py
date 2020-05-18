from PIL import Image
import os
import json
import seaborn as sns
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential
import h5py
from tensorflow.keras.models import model_from_json
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from matplotlib import pyplot as plt
import logging
logging.getLogger('tensorflow').disabled = True


#%%
"""
Creating y 
The authors have not released the y labels of the test set.
Therefore, a new split will be made based on the data that is available (3609 images). 
The original dataset can be found on: https://github.com/gjy3035/NWPU-Crowd-Sample-Code
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
Create model function
We want to try to mimic the parameter settings in from paper. However, we do not have access to the same computational
power. 
In the paper the learning rate is set to 1e-5 (0.00001), batch size = 6, and (max) epochs = 2000.
For the current study, we will limit learning rate to 0.001, batch size 12 and epochs 50. the smaller the learning rate,
the slower the learning. We cannot do this, since we will be doing less epochs. 
Some dropout layers were added to the model. I decided not to add dropout layers after the convolution layers, 
because I have found multiple articles stating that
this is not particularly useful. 
In addition, I have put dropout layers after the input and dense layers. The effect of the dropout layers was very small
I also varies the dropout rate, did not really have an effect. Instead one dropout layer is added after the input. 
The dropout layer after the input did seem to make the model worse. So, I chose to keep the two dropout layers after the
first two dense layers, with a rate of 0.2. Dropout could become more important as we improve the model, since it will 
be more prone to overfitting. 
"""


def create_model(loss_func, learning_rate, dropout):
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
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout))
    # The last layer is size 1, since the output is a continuous value. Also, we do not specify an activation function
    # since it is a regression task and the y-values are not transformed
    model.add(layers.Dense(1))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_func)
    return model


#%%
"""
Making the initial model
"""
# Initialize model and fit model
cnn_50_001 = create_model(loss_func='mean_absolute_error', learning_rate=0.001, dropout=0.2)
cnn_50_001.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_val, y_val))

# Saving the model and weights to disk
cnn_json = cnn_50_001.to_json()
with open("cnn_50_001.json", "w") as json_file:
    json_file.write(cnn_json)
cnn_50_001.save_weights("cnn_50_001.h5")


#%%
# Making Predictions on the regression model
preds_50_001 = cnn_50_001.predict(X_test)
preds_50_001 = np.reshape(preds_50_001, (933,))
cnn_50_001_df = pd.DataFrame({"y_test": y_test, "predictions": preds_50_001})

# Add absolute difference
cnn_50_001_df['abs_difference'] = abs(y_test - preds_50_001)
print("MAE Test:", cnn_50_001_df['abs_difference'].mean())

# Negative samples
negatives_df = cnn_50_001_df.loc[cnn_50_001_df['y_test'] == 0]
print("MAE negative samples:", negatives_df['abs_difference'].mean())
positives_df = cnn_50_001_df.loc[cnn_50_001_df['y_test'] > 0]
print("MAE positive samples:", positives_df['abs_difference'].mean())

# What is the range of predictions
print(f'Lowest prediction: {min(preds_50_001)}, highest prediction {max(preds_50_001)}')

# Train mae: 143.0198, validation mae: 359.9801
# The model performs decently, it does seem to overfit (training loss is quite a bit lower).
# Let's decrease the learning rate


#%%
"""
Adapt learning rate
"""
# We set the learning rate to 0.0001
cnn_50_0001 = create_model(loss_func='mean_absolute_error', learning_rate=0.0001, dropout=0.2)
cnn_50_0001.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_val, y_val))

# Saving the model and weights to disk
cnn_json = cnn_50_0001.to_json()
with open("cnn_50_0001.json", "w") as json_file:
    json_file.write(cnn_json)
cnn_50_0001.save_weights("cnn_50_0001.h5")


#%%
# Making Predictions on the regression model
preds_50_0001 = cnn_50_0001.predict(X_test)
preds_50_0001 = np.reshape(preds_50_0001, (933,))
cnn_50_0001_df = pd.DataFrame({"y_test": y_test, "predictions": preds_50_0001})

# Add absolute difference
cnn_50_0001_df['abs_difference'] = abs(y_test - preds_50_0001)
print("MAE Test:", cnn_50_0001_df['abs_difference'].mean())  # a little better than the previous

# Negative samples
negatives_df = cnn_50_0001_df.loc[cnn_50_0001_df['y_test'] == 0]
print("MAE negative samples:", negatives_df['abs_difference'].mean())
positives_df = cnn_50_0001_df.loc[cnn_50_0001_df['y_test'] > 0]
print("MAE positive samples:", positives_df['abs_difference'].mean())

# What is the range of predictions
print(f'Lowest prediction: {min(preds_50_0001)}, highest prediction {max(preds_50_0001)}')
# The model still never predict a 0
# The model performs decently, it seems to overfit less (training los was 280.4695 in final epoch).
# Training mae: 280.4695, validation mae: 325.3251
# we'll increase epochs and dropout rate (as increasing the epochs will probably also increase overfitting


#%%
"""
Increase epochs and dropout
"""
# We set epochs to 100 and dropout to 0.4 (100 is the maximum amount of epochs without failing)
cnn_100_0001 = create_model(loss_func='mean_absolute_error', learning_rate=0.0001, dropout=0.4)
cnn_100_0001.fit(X_train, y_train, epochs=100, batch_size=12, validation_data=(X_val, y_val))

# Saving the model and weights to disk
cnn_json = cnn_100_0001.to_json()
with open("cnn_100_0001.json", "w") as json_file:
    json_file.write(cnn_json)
cnn_100_0001.save_weights("cnn_100_0001.h5")


#%%
# Making Predictions on the regression model
preds_100_0001 = cnn_100_0001.predict(X_test)
preds_100_0001 = np.reshape(preds_100_0001, (933,))
cnn_100_0001_df = pd.DataFrame({"y_test": y_test, "predictions": preds_100_0001})

# Add absolute difference
cnn_100_0001_df['abs_difference'] = abs(y_test - preds_100_0001)
print("MAE Test:", cnn_100_0001_df['abs_difference'].mean())  # a little better than the previous

# Negative samples
negatives_df = cnn_100_0001_df.loc[cnn_100_0001_df['y_test'] == 0]
print("MAE negative samples:", negatives_df['abs_difference'].mean())
positives_df = cnn_50_0001_df.loc[cnn_100_0001_df['y_test'] > 0]
print("MAE positive samples:", positives_df['abs_difference'].mean())

# What is the range of predictions
print(f'Lowest prediction: {min(preds_100_0001)}, highest prediction {max(preds_100_0001)}')
# The model still does not predict a 0
# The model performs decently, it seems to overfit less (training los was 248.0652 in final epoch), which is surprising
# Increasing the dropout seems to have been a good decision
# Without using feature maps, the model will probably not improve much further. The MAE is okay, however, negative
# examples are never correctly predicted, and the same goes for images with many people.
# Training mae: 257.9869, validation mae: 341.0607


#%%
"""
Visualizing the CNN layers

This code is partially from the lab about CNN's
The code for the plot_layers function is from: 
https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
"""
# Take an example picture form the test set
img_tensor = X_test[[3], :].reshape((1, 256, 256, 3))

# Extract the output of the layers and create a model that will return the outputs
layer_outputs = [layer.output for layer in cnn_100_0001.layers[:7]]
activation_model = models.Model(inputs=cnn_100_0001.input, outputs=layer_outputs)
activation = activation_model.predict(img_tensor)


def plot_layers(model, activations):
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
        # Names of the layers, so you can have them as part of your plot

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


plot_layers(cnn_100_0001, activation)
# We can see the layers becoming more abstract and less interpretable as we go deeper


#%%
"""
Random OverSampling
The random oversampling is only applied on the training data, to prevent data leakage.
# based on paper: https://link.springer.com/article/10.1186/s40537-019-0192-5
# RandomOverSampler documentation (does not mention a maximum amount of dimensions): https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler
"""
# Negative samples
y_train_negatives = [i for i in y_train if i == 0]
print(len(y_train_negatives))

# Let's increase the amount of negative samples by 50% (=211 samples)
ros = RandomOverSampler(sampling_strategy={0: 211}, random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
# Does not work, since the x has to many dimension
# Why is this not mentioned anywhere on the documentation


#%%
"""
Focal loss (classification)
categorical_crossentropy from: https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-function
formula focal loss from paper: https://link.springer.com/article/10.1186/s40537-019-0192-5
defining a custom loss function: https://heartbeat.fritz.ai/how-to-create-a-custom-loss-function-in-keras-637bd312e9ab
"""


def focal_loss(y_true, y_pred):
    """
    Function re-shapes the cross entropy in order to reduce the impact that easily classified samples have on the loss.
    :param y_true:
    :param y_pred:
    :return: focal loss
    """
    alphas_classes = np.array([1, 0.25, 0.25, 0.25])  # Decreases values of class 2,3,4. Minority class stays the same
    alphas_classes = tf.convert_to_tensor(alphas_classes, dtype='float32')
    a = y_true*alphas_classes
    a = tf.reduce_max(a)
    p = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    fl = (-a*(1-p)**2) * tf.math.log(p)
    return fl

# should be applied on a classification task. Let's alter the model.


#%%
# Creating a classification version of the model

# Re-formatting the y
# Inspecting the distribution of our y
print(f"Range of y: ({min(y)}, {max(y)})")
y_temp = [i for i in y if i < 2000]
fig, axx = plt.subplots(1, 2, figsize=(15, 5))
sns.distplot(y, ax=axx[0])
axx[0].set_title('y')
sns.distplot(y_temp, ax=axx[1])
axx[1].set_title('y')
fig.show()

# The data is split into four categories 0 people (negative samples), 0-500, 500-1000 and >1000
y_categorical = []
for value in y:
    if value == 0:
        y_categorical.append(value)
    elif value < 500:
        y_categorical.append(1)
    elif value < 1000:
        y_categorical.append(2)
    else:
        y_categorical.append(3)

print(f"There are {y_categorical.count(0)} negative images")
print(f"There are {y_categorical.count(1)} images with 0-500 people")
print(f"There are {y_categorical.count(2)} images with 500-1000 people")
print(f"There are {y_categorical.count(3)} images with >= 1000 people")


y_categorical = np.asarray(y_categorical)
y_categorical = to_categorical(y_categorical)  # One-hot encode
y_categorical = y_categorical.astype('float32')
# Save y_categorical to disk
pickle.dump(y_categorical, open('y_categorical.pickle', 'wb'))

# Create train, validation and test sets
y_train_cat = y_categorical[:1865]
y_val_cat = y_categorical[1865:2176]
y_test_cat = y_categorical[2176:3109]


#%%
def create_classification_model(loss_func, learning_rate, dropout):
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
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout))
    # The last layer is size 4 since we have four classes
    # Softmax activation is a standard for multi-class classification
    model.add(layers.Dense(4, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # The loss function is adapted for categorical classification and the accuracy metric is added
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'])
    return model


#%%
# We use the categorical crossentropy loss function for the initial classification model
cnn_classifier = create_classification_model(loss_func='categorical_crossentropy', learning_rate=0.0001, dropout=0.4)
cnn_classifier.fit(X_train, y_train_cat, epochs=100, batch_size=12, validation_data=(X_val, y_val_cat))

# Saving the model and weights to disk
cnn_classifier_json = cnn_classifier.to_json()
with open("cnn_classifier.json", "w") as json_file:
    json_file.write(cnn_classifier_json)
cnn_classifier.save_weights("cnn_classifier.h5")


#%%
# load the model
json_file = open('cnn_classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn_classifier = model_from_json(loaded_model_json)
cnn_classifier.load_weights('cnn_classifier.h5')
cnn_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Predicting y on test-set
test_loss, test_acc = cnn_classifier.evaluate(X_test, y_test_cat)
print(test_acc)
preds = cnn_classifier.predict(X_test)
preds_2 = list(np.argmax(preds, axis=1))
print(preds_2)

# The classifier does not perform too bad
y_test_cat = list(np.argmax(y_test_cat, axis=1))
print(f"There are {y_test_cat.count(0)} negative images")
print(f"There are {preds_2.count(0)} negative images predicted")

print(f"There are {y_test_cat.count(1)} images with 0-500 people")
print(f"There are {preds_2.count(1)} images with 0-500 people predicted")

print(f"There are {y_test_cat.count(2)} images with 500-1000 people")
print(f"There are {preds_2.count(2)} images with 500-1000 people predicted")

print(f"There are {y_test_cat.count(3)} images with >= 1000 people")
print(f"There are {preds_2.count(3)} images with >= 1000 people predicted")


# it does overfit. The train accuracy in last epoch was 0,9871. (Its probably more or less just memorizes
# data).


#%%
# Now fit the model with the new loss function
# We use the categorical crossentropy loss function for the initial classification model
cnn_classifier_fl = create_classification_model(loss_func=focal_loss, learning_rate=0.0001, dropout=0.4)
cnn_classifier_fl.fit(X_train, y_train_cat, epochs=100, batch_size=12, validation_data=(X_val, y_val_cat))

# Saving the model and weights to disk
cnn_classifier_json = cnn_classifier.to_json()
with open("cnn_classifier_fl.json", "w") as json_file:
    json_file.write(cnn_classifier_json)
cnn_classifier.save_weights("cnn_classifier_fl.h5")

# The model runs fine, but the focall los is not being calculated
print(len(X))