




class Model:
    def _init_(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )

