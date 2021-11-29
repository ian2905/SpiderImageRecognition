"""
Ian Clouston

TODO:
Notes to self:
    Check out the corrupted image remover code for the server input. found at https://keras.io/examples/vision/image_classification_from_scratch/

Steps left:


Sources/Help from:
    https://keras.io/getting_started/intro_to_keras_for_engineers/
    https://keras.io/examples/vision/image_classification_from_scratch/
"""
import os
import sys

import numpy as np
import tensorflow as tf
import pydot as pydot
import pydotplus as pydotplus
import graphviz as graphviz
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from keras import utils
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

from Resnet50Model import Resnet50Model
from XceptionModel import XceptionModel

"""
Image info
Species: 
Image Size: max 500 HxW
Train Count: 40,687
Validation Count: 1,530

"""

# globals
image_size = (500, 500)
input_size = (350, 350)
shape = (350, 350, 3) # (pixels, pixels, RGB)
edge_size = 350
batch_size = 15
num_classes = 5
epochs = 1
# epochs = 200

xception_model = XceptionModel(shape, num_classes)
resnet50_model = Resnet50Model(shape, num_classes)

#def predict()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# create training and validation datasets from locally stored images
train_ds = keras.preprocessing.image_dataset_from_directory(
    'C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/train5',
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size = input_size,
    batch_size = batch_size,
    labels = "inferred",
    label_mode = 'categorical'
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    'C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/train5',
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size = input_size,
    batch_size = batch_size,
    labels = "inferred",
    label_mode = 'categorical'
)
train_ds = train_ds.prefetch(buffer_size=batch_size)

val_ds = val_ds.prefetch(buffer_size=batch_size)

'''
#for creating numpy arrays
train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory("C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/train5",target_size=(350, 350),batch_size=32,shuffle=True,class_mode='categorical')
test_generator = test_datagen.flow_from_directory("C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/val5",target_size=(350,350),batch_size=32,shuffle=False,class_mode='categorical')
'''

# for data, labels in dataset:
#   print(data.shape)
#   print(data.dtype)
#   print(labels.shape)
#   print(labels.dtype)

# call model function

# Xception model
model = xception_model.make_model()

# resnet-50 model
# model = ResNet50(input_shape=shape)

# es = keras.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)

# model.summmary()

# setup checkpointing
callbacks = [
    # keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),

    # setup early stopping
        # mode: min or max | decreasing or increasing
        # verbose: "1" makes it print out the epoch it stops on
        # patience: number of epochs of stagnation until it stops
    keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20),
]

# train model
history = model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
"""
tuner = keras_tuner.tuners.Hyperband(
  make_modelbasic_tuner,
  objective='val_loss',
  max_epochs=100,
  max_trials=200,
  executions_per_trial=2)
"""

model.save('model.h5')




def load_deep_model(self, model):
    # can specify a directory if you need to
    loaded_model = tf.keras.models.load_model("model.h5")
    return loaded_model

# get a session and save the loaded model to it
# with tf.keras.backend.get_session() as sess:
#     tf.saved_model.simple_save(
#         sess,
#         export_path,
#         inputs={'input_image': model.input},
#         outputs={t.name: t for t in model.outputs})


# -----------------------------------------------------------------------------------------------------------------------
# start

