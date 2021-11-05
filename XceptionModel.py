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

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


class XceptionModel:

    def _init_(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    """
        Description:
            This model is a varient of the Xception network that I found on the Keras main website https://keras.io/examples/vision/image_classification_from_scratch/

            Current val accuracy best: ~55% after 200 epochs, batchsize = 40
            Val accuracy with batch normalization: ~63% after 100 epochs, batchsize = 10(to reduce memory usage), classes=5
    """
    def make_model(self):
        inputs = keras.Input(self.input_shape)

        x = CenterCrop(height=self.input_shape[0], width=self.input_shape[1])(inputs)
        x = data_augmentation(inputs)
        x = layers.Rescaling(1.0 / 255)(x)

        # Entry block
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
        x = layers.Flatten()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", )(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # compile model with optimizer, loss, and metrics
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            # metrics=["accuracy", keras.metrics.SparseCategoricalAccuracy(name="CatAcc")],
        )
        return model