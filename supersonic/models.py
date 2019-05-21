import tensorflow as tf

from supersonic import utils

MODEL_REGISTRY = {}


def model(model_id):
    def register(model_class):
        MODEL_REGISTRY[model_id] = model_class
        return model_class

    return register


def get_model(model_id):
    return MODEL_REGISTRY[model_id]


@model("NatureVision")
class NatureVision(tf.keras.Model):
    """
    The vision half of the network used in the 2015 DQN Nature paper.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            input_shape=(84, 84, 4),
            kernel_size=(8, 8),
            strides=4,
            activation="relu",
            data_format="channels_last",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(4, 4),
            strides=2,
            activation="relu",
            data_format="channels_last",
        )
        self.conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=1,
            activation="relu",
            data_format="channels_last",
        )
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x


@model("NaturePolicy")
class NaturePolicy(tf.keras.Model):
    """
    Standard policy network.
    """

    def __init__(self, nb_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.out = tf.keras.layers.Dense(nb_actions, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x


@model("VanillaValue")
class VanillaValue(tf.keras.Model):
    """
    Standard value network.
    """

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.val = tf.keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.val(x)
        return x


@model("LvlMapVision")
class LvlMapVision(tf.keras.Model):
    """
    A CNN for large, rectangular color images. Will be used for processing lvl maps as part of the task
    picker.
    """

    def __init__(self):
        super().__init__()
        input_shape = utils.get_avg_lvl_map_dims() + (3,)
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            input_shape=input_shape,
            kernel_size=(16, 16),
            strides=6,
            activation="relu",
            data_format="channels_last",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(4, 4),
            strides=2,
            activation="relu",
            data_format="channels_last",
        )
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        return x


@model("LvlMapPolVal")
class LvlMapPolVal(tf.keras.Model):
    def __init__(self):
        super().__init__()
        output_shape = len(utils.all_sonic_lvls().keys())
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.out = tf.keras.layers.Dense(output_shape, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.out(x)
        return x
