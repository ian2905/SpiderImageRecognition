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

# randomly transform images to reduce overfitting
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

# resnet-50 conv block



class Resnet50Model:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __convolutional_block(self, x, f, filters, stage, block, s=2):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        x_shortcut = x

        x = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                          kernel_initializer=glorot_uniform(seed=0))(x)
        x = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                          kernel_initializer=glorot_uniform(seed=0))(x)
        x = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                          kernel_initializer=glorot_uniform(seed=0))(x)
        x = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        x_shortcut = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                   name=conv_name_base + '1',
                                   kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
        x_shortcut = layers.BatchNormalization(axis=3, name=bn_name_base + '1')(x_shortcut)

        x = layers.Add()([x, x_shortcut])
        x = layers.Activation('relu')(x)

        return x

    # resnet-50
    # found and learned about through this tutorial https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
    # identiy block for resnet-50
    def __identity_block(self, x, f, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters

        x_shortcut = x

        x = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                          kernel_initializer=glorot_uniform(seed=0))(x)
        x = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                          kernel_initializer=glorot_uniform(seed=0))(x)
        x = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                          kernel_initializer=glorot_uniform(seed=0))(x)
        x = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        x = layers.Add()([x, x_shortcut])  # SKIP Connection
        x = layers.Activation('relu')(x)

        return x



    def make_model(self):
        x_input = keras.Input(self.input_shape)

        # preprocessing by cropping image, applying random transformations, and rescalling output to between 0 and 1
        x = CenterCrop(height=self.input_shape[0], width=self.input_shape[1])(x_input)
        x = data_augmentation(x_input)
        x = layers.Rescaling(1.0 / 255)(x)

        x = layers.ZeroPadding2D((3, 3))(x_input)

        x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(x)
        x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.__convolutional_block(x, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        x = self.__identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.__identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.__convolutional_block(x, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        x = self.__identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.__identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.__identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.__convolutional_block(x, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.__convolutional_block(x, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        x = self.__identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.__identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        # ...alternatively...
        # x = layers.GlobalAveragePooling2D()(x)

        # TODO: add dropout and test training
        x = layers.Dropout(0.5)(x)

        base_model = keras.Model(inputs=x_input, outputs=x, name='ResNet50')

        # apply transfer learning
        headModel = base_model.output
        headModel = layers.Flatten()(headModel)
        headModel = layers.Dense(256, activation='relu', name='fc1', kernel_initializer=glorot_uniform(seed=0))(
            headModel)
        headModel = layers.Dense(128, activation='relu', name='fc2', kernel_initializer=glorot_uniform(seed=0))(
            headModel)
        headModel = layers.Dense(self.num_classes, activation='softmax', name='fc3',
                                 kernel_initializer=glorot_uniform(seed=0))(headModel)

        model = keras.Model(inputs=base_model.input, outputs=headModel)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            # metrics=["accuracy", keras.metrics.SparseCategoricalAccuracy(name="CatAcc")],
        )

        return model