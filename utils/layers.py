# Author:  DINDIN Meryll
# Date:    15/03/2019
# Project: ml_utils

try: from ml_utils.utils import *
except: from utils import *

# Callback designed to work in synergy with the Adaptive Dropout layer

class DecreaseDropout(Callback):

    def __init__(self, prb, steps):

        super(Callback, self).__init__()

        self.ini = prb
        self.prb = prb
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):

        self.prb = max(0, 1 - epoch/self.steps) * self.ini

# Defines the Adaptive Dropout layer based on callback feedback

class AdaptiveDropout(Layer):

    def __init__(self, p, callback, **kwargs):

        self.p = p
        self.callback = callback
        if 0. < self.p < 1.: self.uses_learning_phase = True
        self.supports_masking = True

        super(AdaptiveDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):

        self.p = self.callback.prb

        if 0. < self.p < 1.:
            x = K.in_train_phase(K.dropout(x, level=self.p), x)

        return x

    def get_config(self):

        config = {'p': self.p}
        base_config = super(AdaptiveDropout, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# Built Attention layer, to be used in certain relevant circumstances

class Attention(Layer):
    
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        
        self.supports_masking = True
        self.init = initializers.get('he_uniform')
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        assert len(input_shape) == 3

        a_0 = {'initializer': self.init, 'name': '{}_W'.format(self.name)}
        a_1 = {'regularizer': self.W_regularizer, 'constraint': self.W_constraint}
        self.W = self.add_weight((input_shape[-1],), **a_0, **a_1)
        self.features_dim = input_shape[-1]

        if self.bias:
            a_0 = {'initializer': 'zero', 'name': '{}_b'.format(self.name)}
            a_1 = {'regularizer': self.b_regularizer, 'constraint': self.b_constraint}
            self.b = self.add_weight((input_shape[1],), **a_0, **a_1)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        
        return None

    def call(self, x, mask=None):
        
        u = K.reshape(x, (-1, self.features_dim))
        v = K.reshape(self.W, (self.features_dim, 1))
        eij = K.reshape(K.dot(u, v), (-1, self.step_dim))
        if self.bias: eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None: a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        
        return input_shape[0],  self.features_dim
