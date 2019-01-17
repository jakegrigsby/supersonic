import tensorflow as tf

class NatureVision(tf.keras.Model):
    """
    The vision half of the network used in the 2015 DQN Nature paper.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_last')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_last')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_last')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x

class NaturePolicy(tf.keras.Model):
    """
    Standard policy network.
    """
    def __init__(self, nb_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.out = tf.keras.layers.Dense(nb_actions, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.out(x)
        return x


class VanillaValue(tf.keras.Model):
    """
    Standard value network.
    """
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.val = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.val(x)
        return x
