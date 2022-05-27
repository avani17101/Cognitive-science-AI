
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
from Models.encoder import *
from Models.decoder import *

def encdec(NUM_VOXELS,RESOLUTION,encoder_model,decoder_model):
    '''
    Encoder is pre-trained. Decoder is trained in two settings:
     1) encoder-decoder: stack encoder followed by decoder and do training of decoder
     2) decoder-encoder: stack decoder followed by encoder and do training of decoder
    '''
    voxel_inp = Input((NUM_VOXELS,))
    img_inp = Input((RESOLUTION, RESOLUTION, 3))
    mode_inp = Input((1,))

    voxel_pred = encoder_model(img_inp)
    rec_img_dec = decoder_model(voxel_inp)

    rec_img_encdec = decoder_model(voxel_pred)
    voxel_pred_decenc = encoder_model(rec_img_dec)

    recons_im_out = Lambda(lambda i: K.switch(i[0],i[1],i[2]) ,name = 'recons_im_out') ([mode_inp,rec_img_dec,rec_img_encdec])  #switching based on mode
    out_voxel_pred = Lambda(lambda i: K.switch(i[0], i[1], i[2]), name='out_voxel_pred')(
            [mode_inp, voxel_pred, voxel_pred_decenc])

    return Model(inputs=[voxel_inp,img_inp,mode_inp],outputs=[recons_im_out,out_voxel_pred]) #,out_voxel_pred