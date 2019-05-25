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
            activation="linear",
            data_format="channels_last",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(4, 4),
            strides=2,
            activation="linear",
            data_format="channels_last",
        )
        self.conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=1,
            activation="linear",
            data_format="channels_last",
        )
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.dense1 = tf.keras.layers.Dense(
            256,
            activation="linear",
        )
        self.dense2 = tf.keras.layers.Dense(
            448,
            activation="linear",
        )
        self.dense3 = tf.keras.layers.Dense(
            128,
            activation="linear",
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.dense2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.dense3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x


@model("ExplorationTarget")
class ExplorationTarget(tf.keras.Model):
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
        self.dense1 = tf.keras.layers.Dense(128, activation="linear")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x


@model("ExplorationTrain")
class ExplorationTrain(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            input_shape=(84, 84, 4),
            kernel_size=(8, 8),
            strides=4,
            activation="linear",
            data_format="channels_last",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(4, 4),
            strides=2,
            activation="linear",
            data_format="channels_last",
        )
        self.conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=1,
            activation="linear",
            data_format="channels_last",
        )
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.dense3 = tf.keras.layers.Dense(128, activation="linear")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


@model("VanillaPolicy")
class VanillaPolicy(tf.keras.Model):
    """
    Standard policy network.
    """

    def __init__(self, nb_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(
            256, activation="linear",
        )
        self.out = tf.keras.layers.Dense(
            nb_actions,
            activation="softmax",
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.out(x)
        return x


@model("VanillaValue")
class VanillaValue(tf.keras.Model):
    """
    Standard value network.
    """

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(
            128, activation="relu"
        )
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.val = tf.keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs) + inputs
        x = self.dense2(x)
        x = self.val(x)
        return x
