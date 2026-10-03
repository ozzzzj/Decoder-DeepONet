
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


class ResidualBlock_dense(tf.keras.layers.Layer):
    def __init__(self, units, l2_lambda=0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.l2_lambda = l2_lambda
        self.dense1 = tf.keras.layers.Dense(units, activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_lambda))
        # self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(units, activation=None,
            kernel_regularizer=regularizers.l2(self.l2_lambda))
        # self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.act = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        # x = self.bn1(x)
        x = self.dense2(x)
        # x = self.bn2(x)
        x = self.add([x, inputs])
        x = self.act(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "l2_lambda": self.l2_lambda
        })
        return config


# ---------------- Jitter layer ----------------
class Jitter(tf.keras.layers.Layer):
    def __init__(self, stddev=0.05, prob=0.8, **kw):
        super().__init__(**kw)
        self.stddev = stddev
        self.prob   = prob

    def call(self, x, training=None):
        if not training:
            return x
        mask = tf.less(tf.random.uniform((tf.shape(x)[0], 1, 1)), self.prob)
        noise = tf.random.normal(tf.shape(x), stddev=self.stddev)
        return tf.where(mask, x + noise, x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'stddev': self.stddev, 'prob': self.prob})
        return cfg