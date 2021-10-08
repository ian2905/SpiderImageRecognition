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
input_size = (350, 350)
batch_size = 40
num_classes = 5
#epochs = 50
epochs = 200


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def make_modelbasic(shape, num_classes):
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
    return keras.Model(inputs=inputs, outputs=outputs)

def make_model2(shape, num_classes):

    inputs = keras.Input(shape)

    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", )(x)
    return keras.Model(inputs=inputs, outputs=outputs)

#create training and validation datasets from locally stored images
train_ds = keras.preprocessing.image_dataset_from_directory(
    'C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/train5',
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size = input_size,
    batch_size = batch_size,
    label_mode = 'categorical'
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    'C:/Users/clous/OneDrive/Desktop/CS/CS_Proj/Spider/Images/INat/train5',
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size = input_size,
    batch_size = batch_size,
    label_mode = 'categorical'
)

#randomly transform images to reduce overfitting
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

train_ds = train_ds.prefetch(buffer_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=batch_size)

#for data, labels in dataset:
#   print(data.shape)
#   print(data.dtype)
#   print(labels.shape)
#   print(labels.dtype)

#call model function
model = make_modelbasic(shape=(350, 350, 3), num_classes=num_classes)

#model.summmary()

#setup checkpointing
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

#compile model with optimizer, loss, and metrics
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy". keras.metrics.SparseCategoricalAccuracy(name="CatAcc")],
)

#train model
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)


