
import os
import configs.config as config
import math
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import cosine_similarity


from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from utils.loss_utils import *
from Models.encdec_model import *
from utils.data_loader_utils import *
from utils.misc_utils import *



os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_encoder_decoder_and_rev(config_file,encoder_model):
    '''
    decoder is trained as mentioned in report
    '''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    data_loader = data_handler(matlab_file = config_file.kamitani_data_mat)

    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


    Y,Y_test,Y_test_avg = data_loader.get_data(roi = 'ROI_VC',imag_data = 0)
    labels_train, labels = data_loader.get_labels(imag_data = 0)

    data_imgs= np.load(config_file.images_npz) 
    X ,X_test_avg = data_imgs['train_images'][labels_train], data_imgs['test_images']
    X_test = X_test_avg[labels]

    NUM_VOXELS = Y.shape[1]

    Tv_reg =1
    image_loss_ = image_loss()

    
    snr  = snr_calc(Y_test,Y_test_avg,labels)
    SNR  = tf.constant(snr/snr.mean(),shape = [1,len(snr)],dtype = tf.float32)

    def feature_loss(y_true, y_pred ):
        # print("inside feature_loss",y_true.shape, y_pred.shape)
        l = config_file.w1*image_loss_.vgg_loss(y_true, y_pred,'block2_conv2')
        l +=config_file.w2*image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')
        l +=config_file.w3*image_loss_.pixel_loss(y_true, y_pred)
        return l
        #return image_loss_.pixel_loss(y_true, y_pred)#image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')

    def combined_loss(y_true, y_pred):
        # print("TRUE PRED",y_true.shape,y_true, y_pred.shape)
        return feature_loss(y_true, y_pred)+  Tv_reg *total_variation_loss(y_pred)

    def mae_vox(y_true, y_pred):
        # print("inside mae_vox",y_true.shape, y_pred.shape)
        ar = SNR*K.abs(y_true-y_pred)
        return K.mean(ar,axis=-1)

    def combined_voxel_loss(y_true, y_pred):
        l = config_file.w_mae_vox*mae_vox(y_true, y_pred)
        l += config_file.w_cs_vox *cosine_similarity(y_true, y_pred)
        return l

    if not os.path.exists(config_file.results):
        os.makedirs(config_file.results)

    epochs_drop =  5.0
    initial_lrate = 0.001
    
    RESOLUTION = config_file.image_size
    

    def step_decay(epoch):
        drop = 0.2
        lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate

    image_loss_.calc_norm_factors(X)
    decoder_model = decoder(NUM_VOXELS)

    model = encdec(NUM_VOXELS,RESOLUTION,encoder_model,decoder_model)
    model.compile(loss= {'out_rec_img':combined_loss,'out_pred_voxel':combined_voxel_loss},loss_weights=[1.0,1.0],optimizer=tf.keras.optimizers.Adam(lr=5e-4,amsgrad=True),metrics={'out_rec_img':['mse','mae']})

    lis_callbacks = []
    lis_callbacks.append(LearningRateScheduler(step_decay))
    save_test_imgs = log_save_imgs_in_collage_callback(Y_test_avg, X_test_avg, decoder_model, dir = config_file.results+'/test_collge_ep/')
    lis_callbacks.append(save_test_imgs)
    save_train_imgs = log_save_imgs_in_collage_callback(Y[0:50], X[0:50], decoder_model, dir = config_file.results+'/train_collge_ep/')
    lis_callbacks.append(save_train_imgs)

    loader_train = encoder_dec_dataloader(X, Y, Y_test, labels, batch_paired = 48, batch_unpaired = 16)

    model.fit_generator(loader_train, epochs=config_file.decoder_epochs, verbose=2,callbacks=lis_callbacks,workers=1) #epochs
    save_imgs_in_collage([X_test_avg,decoder_model.predict(Y_test_avg)], rows =10, border =5,save_file = config_file.results+'/collage.jpeg')
    y_test_pred = decoder_model.predict(Y_test_avg)
    y_train_pred = decoder_model.predict(Y[0:50])
    
    save_images(y_test_pred,images_orig = X_test_avg ,folder=config_file.results+'/test/')
    save_images(y_train_pred,images_orig = X[0:50] ,folder=config_file.results+'/train/')

    if(config_file.decoder_weights is not None):
        decoder_model.save_weights(config_file.decoder_weights)