
import os
import math
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.losses import cosine_similarity

from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from utils.misc_utils import snr_calc
from utils.misc_utils import *

from utils.loss_utils import *
from utils.data_loader_utils import *
from Models.encdec_model import *

def testing(config_file):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


    #################################################### data load #########################################################
    handler = data_handler(matlab_file = config_file.kamitani_data_mat)
    Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0)
    labels_train, labels = handler.get_labels(imag_data = 0)
    
    data_imgs= np.load(config_file.images_npz) 
    X ,X_test_avg = data_imgs['train_images'][labels_train], data_imgs['test_images']
    X_test = X_test_avg[labels]

    
    NUM_VOXELS = Y.shape[1]
    #################################################### losses ##########################################################

    snr  = snr_calc(Y_test,Y_test_avg,labels)
    SNR  = tf.constant(snr/snr.mean(),shape = [1,len(snr)],dtype = tf.float32)

 
    Tv_reg =1
    image_loss_ = image_loss()

    def feature_loss(y_true, y_pred ):
        # print("inside feature_loss",y_true.shape, y_pred.shape)
        return 0.15*image_loss_.vgg_loss(y_true, y_pred,'block2_conv2')+0.7*image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')+0.15*image_loss_.pixel_loss(y_true, y_pred)
        #return image_loss_.pixel_loss(y_true, y_pred)#image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')

  
    #################################################### learning param & schedule #########################################


    initial_lrate = 0.001
    epochs_drop =  30.0
    RESOLUTION = config_file.image_size
  
    ##################################################### model ############################################################
    image_loss_.calc_norm_factors(X)
    decoder_weights = config_file.decoder_weights
    decoder_model = decoder(NUM_VOXELS)
    decoder_model.load_weights(decoder_weights )
    preds = decoder_model.predict(Y_test_avg)


    save_imgs_in_collage([X_test_avg,preds], rows =10, border =5,save_file = config_file.results+'/collage_new.jpeg')
    save_images(preds,images_orig = X_test_avg ,folder=config_file.results+'/test_new/')
    print("saved images to ", config_file.results)