
import tensorflow as tf
import tensorflow.keras.backend as K

from supersonic import utils

"""
As part of the plan to minimize MPI complexity, most of the meta-learning/param-search algs will involve passing strings
and numpy arrays rather than Python objects. So these neural nets should all be able to be initialized with a single string.
The global variable MODEL_REGISTRY handles that. After a network is defined using the keras.Model subclassing API, use the
@model('name') decorator to add it to MODEL_REGISTRY. Then any model can be initialized by calling get_model('name')()
"""

MODEL_REGISTRY = {}
def model(model_id):
    def register(model_class):
        MODEL_REGISTRY[model_id] = model_class
        return model_class
    return register

def get_model(model_id):
    return MODEL_REGISTRY[model_id]

@model('NatureVision')
class NatureVision(tf.keras.Model):
    """
    The vision half of the network used in the 2015 DQN Nature paper.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, input_shape=(84,84,4), kernel_size=(8,8), strides=4, activation='relu', data_format='channels_last')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_last')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_last')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x

@model('NaturePolicy')
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

@model('VanillaValue')
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

@model('LvlMapVision')
class LvlMapVision(tf.keras.Model):
    """
    A CNN for large, rectangular color images. Will be used for processing lvl maps as part of the task
    picker.
    """
    def __init__(self):
        super().__init__()
        input_shape = utils.get_avg_lvl_map_dims() + (3,)
        self.conv1 = tf.keras.layers.Conv2D(32, input_shape=input_shape, kernel_size=(16,16), strides=6, activation='relu', data_format='channels_last')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_last')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        return x

@model('LvlMapPolVal')
class LvlMapPolVal(tf.keras.Model):
    def __init__(self):
        super().__init__()
        output_shape = len(utils.all_sonic_lvls().keys())
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(output_shape, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.out(x)
        return x

@model('NoisyQNetwork')
class NoisyQNetwork(tf.keras.Model):
    def __init__(self, nb_action):
        super().__init__()
        self.vision = NatureVision()
        self.dense = NoisyNetDense(512, activation='relu')
        self.dense2 = NoisyNetDense(nb_action + 1, activation='linear')
        self.out = tf.keras.layers.Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))
    
    def call(self, inputs):
        x = self.vision(inputs)
        x = self.dense(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class NoisyNetDense(tf.keras.layers.Layer):
    """
    A modified fully-connected layer that injects noise into the parameter distribution
    before each prediction. This randomness forces the agent to explore - at least
    until it can adjust its parameters to learn around it.

    To use: replace Dense layers (like the classifier at the end of a DQN model)
    with NoisyNetDense layers and set your policy to GreedyQ.
    See examples/noisynet_pdd_dqn_atari.py

    Reference: https://arxiv.org/abs/1706.10295
    """
    def __init__(self,
                units,
                activation=None,
                kernel_constraint=None,
                bias_constraint=None,
                kernel_regularizer=None,
                bias_regularizer=None,
                mu_initializer=None,
                sigma_initializer=None,
                **kwargs):

        super(NoisyNetDense, self).__init__(**kwargs)

        self.units = units

        self.activation = tf.keras.activations.get(activation)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint) if kernel_constraint is not None else None
        self.bias_constraint = tf.keras.constraints.get(bias_constraint) if kernel_constraint is not None else None
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)if kernel_constraint is not None else None
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer) if kernel_constraint is not None else None

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        #See section 3.2 of Fortunato et al.
        sqr_inputs = self.input_dim.value**(1/2)
        self.sigma_initializer = tf.keras.initializers.Constant(value=.5/sqr_inputs)
        self.mu_initializer = tf.keras.initializers.RandomUniform(minval=(-1/sqr_inputs), maxval=(1/sqr_inputs))


        self.mu_weight = self.add_weight(shape=(self.input_dim, self.units),
                                        initializer=self.mu_initializer,
                                        name='mu_weights',
                                        constraint=self.kernel_constraint,
                                        regularizer=self.kernel_regularizer)

        self.sigma_weight = self.add_weight(shape=(self.input_dim, self.units),
                                        initializer=self.sigma_initializer,
                                        name='sigma_weights',
                                        constraint=self.kernel_constraint,
                                        regularizer=self.kernel_regularizer)

        self.mu_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.mu_initializer,
                                        name='mu_bias',
                                        constraint=self.bias_constraint,
                                        regularizer=self.bias_regularizer)

        self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.sigma_initializer,
                                        name='sigma_bias',
                                        constraint=self.bias_constraint,
                                        regularizer=self.bias_regularizer)

        super(NoisyNetDense, self).build(input_shape=input_shape)

    def call(self, x):
        #sample from noise distribution
        e_i = K.random_normal((self.input_dim, self.units))
        e_j = K.random_normal((self.units,))

        #We use the factorized Gaussian noise variant from Section 3 of Fortunato et al.
        eW = K.sign(e_i)*(K.sqrt(K.abs(e_i))) * K.sign(e_j)*(K.sqrt(K.abs(e_j)))
        eB = K.sign(e_j)*(K.abs(e_j)**(1/2))

        #See section 3 of Fortunato et al.
        noise_injected_weights = K.dot(x, self.mu_weight + (self.sigma_weight * eW))
        noise_injected_bias = self.mu_bias + (self.sigma_bias * eB)
        output = K.bias_add(noise_injected_weights, noise_injected_bias)
        if self.activation != None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'mu_initializer': initializers.serialize(self.mu_initializer),
            'sigma_initializer': initializers.serialize(self.sigma_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(NoisyNetDense, self).get_config()

        return dict(list(bas_config.items()) + list(config.items()))