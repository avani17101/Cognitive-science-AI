
from tensorflow.keras.layers import  Conv2D,Reshape,UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1
import tensorflow as tf
from Models.caffenet_model import *
from Models.layers import *
import configs.config as config


def decoder(num_voxels,name = 'decoder'):
    '''
    model for decoder defined here
    '''
    num_voxels = num_voxels
    num_conv_layers = 3
    input_shape = (num_voxels,)
    model = Sequential(name = name)
    model.add(dense_f2c_gl( out=[14,14,64],l1=20,gl=400,gl_n=0.5,input_shape = input_shape,name='fc_d') )
    model.add(Reshape((14, 14, 64)))
    for i in range(num_conv_layers):
        model.add(Conv2D( 64, (3, 3), padding='same',kernel_initializer="glorot_normal", activation='relu',kernel_regularizer=l1(1e-5 )))
        model.add(UpSampling2D((2, 2))) #,interpolation='bilinear'
        model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(3, (3, 3), padding='same', kernel_initializer="glorot_normal",activation='sigmoid',kernel_regularizer=l1( 1e-5 )))
    return model
