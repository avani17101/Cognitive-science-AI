
from keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.regularizers import l2,l1_l2,l1
'''
referred from https://github.com/WeizmannVision/ssfmri2im
'''
class GroupLasso(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        gl: Float; gl group regularization factor.
        gl_n: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0., gl=0.,gl_n=0):
        self.l1 = K.cast_to_floatx(l1)
        self.gl = K.cast_to_floatx(gl)
        self.gl_n = K.cast_to_floatx(gl_n)


    def __call__(self, x):

        regularization = self.l1 * K.mean(K.abs(x))
        x_sq = K.square(x)

        if (self.gl_n > 0):
            x_sq_pad = tf.pad(x_sq, [[1, 1], [1, 1], [0, 0], [0, 0]], "SYMMETRIC")
            x_sq_avg_n = (x_sq_pad[:-2, 1:-1] + x_sq_pad[2:, 1:-1] + x_sq_pad[1:-1, :-2] + x_sq_pad[1:-1, 2:]) / 4
            x_sq = (x_sq + self.gl_n * x_sq_avg_n) / (1 + self.gl_n)
        regularization += self.gl * K.mean(K.sqrt(K.mean(x_sq, axis=-2)))  # assumes weight structure x,y,ch,voxel
        return regularization


    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.gl)}


def list_prod(l):
    prod = 1
    for e in l:
        prod*=e
    return prod



class SwitchLayer(Layer):
    def __init__(self, **kwargs):
        super(SwitchLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SwitchLayer, self).build(input_shape)
        self.trainable = False


    def call(self, inputs):
        return K.switch(tf.constant(1), inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]



class dense_c2f_gl(Layer):

    def __init__(self, units=1024,l1=0.1,gl=0.1,gl_n=0,kernel_init = "glorot_normal", **kwargs):
        self.units = units
        self.l1 = l1
        self.gl = gl
        self.gl_n = gl_n
        self.kernel_init =kernel_init
        super(dense_c2f_gl, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = list(input_shape[1:])+[self.units]
        self.kernel = self.add_weight(name='dense_c2f_gl_kernel',
                                      shape=shape,
                                      regularizer= GroupLasso(l1=self.l1,gl=self.gl,gl_n= self.gl_n) ,

                                      initializer=self.kernel_init,
                                      trainable=True)
        self.bias = self.add_weight(name='dense_c2f_gl_bias', shape=(self.units,),initializer="glorot_normal",trainable=True)

        super(dense_c2f_gl, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        input_shape = self.s
        size =  list_prod(input_shape[1:])
        x = tf.reshape(x, [-1, size])
        w = tf.reshape(self.kernel,[size ,self.units])
        output = K.dot(x, w)

        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)



class dense_f2c_gl(Layer):

    def __init__(self, out=[4,4,4],l1=0.1,gl=0.1,gl_n= 0.0,kernel_init = "glorot_normal", **kwargs):
        self.units = 1
        self.l1 = l1
        self.gl = gl
        self.gl_n = gl_n
        self.out = out
        self.kernel_init = kernel_init
        super(dense_f2c_gl, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = self.out+list(input_shape[1:])
        self.kernel = self.add_weight(name='dense_f2c_gl_kernel',
                                      shape=shape,
                                      regularizer= GroupLasso(l1=self.l1,gl=self.gl,gl_n=self.gl_n),
                                      initializer=self.kernel_init,
                                      trainable=True)
        self.bias = self.add_weight(name='dense_f2c_gl_bias', shape=(self.out),initializer="glorot_normal",trainable=True)
        super(dense_f2c_gl, self).build(input_shape)


    def call(self, x):
        input_shape = self.s
        len =  input_shape[1]
        w = tf.transpose(self.kernel,perm=[3,0,1,2])
        w = tf.reshape(w,[len,-1])
        output = K.dot(x, w)
        output = tf.reshape(output,[-1]+self.out)
        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]]+ self.out)


