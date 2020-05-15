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
    return n_humans


y = retrieve_y('C:/Users/jarno/Documents/MDSE/Semester 2/Deep Learning/Project/NWPU-Crowd/jsons')
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
    return images


X = resizing_images('C:/Users/jarno/Documents/MDSE/Semester 2/Deep Learning/Project/NWPU-Crowd/images',  256)
pickle.dump(X, open('X.pickle', 'wb'))

# Let´s look at a resized image
img = Image.fromarray(X[0], 'RGB')
print(img.size)
img.show()

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
Initial CNN
Perfmance metrics: MAE, MSE
"""
model = Sequential()
# stage 1 convulutions
# stage 2 acitvations through nonlinear activation function
# stage 3 pooling to modify output


# Use random search to tune hyperparameters?
