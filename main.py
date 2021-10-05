"""
Ian Clouston

TODO:
Notes to self:
    Check out the corrupted image remover code for the server input. found at https://keras.io/examples/vision/image_classification_from_scratch/

Sources/Help from:
    https://keras.io/getting_started/intro_to_keras_for_engineers/
    https://keras.io/examples/vision/image_classification_from_scratch/
"""







import numpy as np
import tensorflow as tf
import pydot as pydot
import pydotplus as pydotplus
import graphviz as graphviz
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from keras import utils
import matplotlib.pyplot as plt


"""
Image info
Species: 
Image Size: max 500 HxW
Train Count: 40,687
Validation Count: 1,530

"""

#globals
image_size = (500, 500)
batch_size = 40
num_classes = 5
epochs = 50;


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

def make_model(shape, num_classes):
    inputs = keras.Input(shape)

    x = CenterCrop(height=350, width=350)(inputs)
    x = data_augmentation(inputs)
    x = Rescaling(scale=(1.0 / 255))(x)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)



train_ds = keras.preprocessing.image_dataset_from_directory(
    'C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/train5',
    image_size = image_size,
    batch_size = batch_size
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    'C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/val5',
    image_size = image_size,
    batch_size = batch_size
)

train_ds = train_ds.prefetch(buffer_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=batch_size)

#for data, labels in dataset:
#   print(data.shape)
#   print(data.dtype)
#   print(labels.shape)
#   print(labels.dtype)

inputs = keras.Input(shape=(350, 350, 3))

x = CenterCrop(height = 350, width = 350)(inputs)
x = data_augmentation(inputs)
x = Rescaling(scale = (1.0 / 255))(x)

x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

#model.summmary()