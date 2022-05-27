
from tensorflow.keras.layers import Input, Conv2D, Lambda,  Flatten, Dropout,Reshape,UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2,l1_l2,l1
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add
from tensorflow.keras.activations import relu
from tensorflow.keras import layers
import tensorflow as tf
from Models.caffenet_model import *
from Models.layers import *
import configs.config as config

MEAN_PIXELS = [123.68, 116.779, 103.939]

def subtract_mean(x):
    mean = tf.constant(MEAN_PIXELS,shape=[1,1,1,3],dtype=tf.float32)
    return tf.subtract(x,mean)

def encoder(num_voxels,name = 'encoder'):
    '''
    model for encoder defined here
    '''
    num_voxels = num_voxels
    resolution = config.image_size
    num_conv_layers = 2
    drop_out = 0.5

    caffenet_models_weights =  scipy.io.loadmat(config.caffenet_models_weights)
    caffenet_models_weights =  caffenet_models_weights['layers']

    input_shape = (resolution, resolution, 3)
    # adding layers
    model = Sequential(name =name)
    model.add(Lambda(lambda img: img[:, :, :, ::-1] * 255.0, input_shape=input_shape))
    model.add(Lambda(subtract_mean))
    model.add(Lambda(conv2d_relu_, arguments={'net_layers': caffenet_models_weights, 'layer': 0, 'layer_name': 'conv1', 'stride': 2, 'pad': 'SAME'}))
    model.add(BatchNormalization(axis=-1))
    for i in range(num_conv_layers):
        model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer="glorot_normal", activation='relu',
                                kernel_regularizer=l1_l2(1e-5,0.001), strides=(2, 2)))

        model.add(BatchNormalization(axis=-1))

    model.add(Flatten())  # flatten needed for dropout, without it suboptimal results are observed
    model.add(Dropout(drop_out))

    model.add(Reshape((14, 14, 32)))
    model.add(dense_c2f_gl(units=num_voxels, l1=10, gl=800, gl_n=0.5))
    return model