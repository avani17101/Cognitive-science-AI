
import os
import math
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.losses import cosine_similarity

from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from utils.misc_utils import snr_calc, log_save_imgs_in_collage_callback
from utils.misc_utils import *

from scipy import stats as stat

from utils.loss_utils import *
from utils.data_loader_utils import *
from Models.encdec_model import *
import configs.config as config_file

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


handler = data_handler(matlab_file = config_file.kamitani_data_mat)
Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0)
labels_train, labels = handler.get_labels(imag_data = 0)

data_imgs= np.load(config_file.images_npz) 
X ,X_test_avg = data_imgs['train_images'][labels_train], data_imgs['test_images']
X_test = X_test_avg[labels]

NUM_VOXELS = Y.shape[1]

snr  = snr_calc(Y_test,Y_test_avg,labels)
SNR  = tf.constant(snr/snr.mean(),shape = [1,len(snr)],dtype = tf.float32)


def mae_vox(y_true, y_pred):
    # print("inside mae_vox",y_true.shape, y_pred.shape)
    ar = SNR*K.abs(y_true-y_pred)
    return K.mean(ar,axis=-1)

def combined_voxel_loss(y_true, y_pred):
        l = config_file.w_mae_vox*mae_vox(y_true, y_pred)
        l += config_file.w_cs_vox *cosine_similarity(y_true, y_pred)
        return l
        
def maelog_vox(y_true, y_pred):
    return K.mean(SNR*K.log(K.abs(y_true-y_pred)+1),axis=-1)


Tv_reg =1
image_loss_ = image_loss()

def feature_loss(y_true, y_pred ):
        # print("inside feature_loss",y_true.shape, y_pred.shape)
        l = config_file.w1*image_loss_.vgg_loss(y_true, y_pred,'block2_conv2')
        l +=config_file.w2*image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')
        l +=config_file.w3*image_loss_.pixel_loss(y_true, y_pred)
        return l

def combined_loss(y_true, y_pred):
    # print("TRUE PRED",y_true.shape,y_true, y_pred.shape)
    return feature_loss(y_true, y_pred)+  Tv_reg *total_variation_loss(y_pred)


initial_lrate = 0.001
epochs_drop =  30.0
RESOLUTION = config_file.image_size
epochs = int(epochs_drop*5)
examples = 16
include_decenc= 1
frac = 3

def step_decay(epoch):

   drop = 0.2
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate



##################################################### model ############################################################
image_loss_.calc_norm_factors(X)

decoder_weights = config_file.decoder_weights
decoder_model = decoder(NUM_VOXELS)
decoder_model.load_weights(decoder_weights )

def calc_metrics(preds, gt_lis):
    mse_lis = []
    mae_lis = []
    for gt,pred in zip(gt_lis,preds):
        mse = np.square(np.subtract(gt,preds)).mean()
        mse_lis.append(mse)
        mae = np.abs(np.subtract(gt,preds)).mean()
        mae_lis.append(mae)

#     save_imgs_in_collage([gt_lis,preds], rows =10, border =5,save_file = config_file.results+'/collage.jpeg')
#     save_images(preds,images_orig = gt_lis ,folder=config_file.results+'/test/')

    mse_lis = np.array(mse_lis)
    mae_lis = np.array(mae_lis)



    mse_lis.mean()

    print("mse {} mae {}".format(mse_lis.mean(),mae_lis.mean()))

    def display_img_arr(img_arr, r, c, dim,titles_arr):
        fl = 0
        fig = plt.figure(figsize = dim)
        for i in range(r):
            for j in range(c):
                if len(img_arr) == fl:
                    break
                ax1 = fig.add_subplot(r, c, fl + 1)
                ax1.set_title(titles_arr[fl], fontsize = 20)
                ax1.imshow(img_arr[fl], cmap = 'gray')
                fl = fl + 1
        plt.show()

    cors = []
    #n-way
    def calc_identification_accuracy(nWay,runs=20):
        correct = 0
        wrong = 0
        for r in range(runs):
            for k,(gt,pred) in enumerate(zip(gt_lis,preds)):
                cor_lis = []
    #             if k==0 and r==0:
    #                 display_img_arr([gt,pred], 1,2,(16,40), ['gt','pred'])

                for i in range(nWay):
                    img = random.choice(gt_lis)
                    cor = stat.pearsonr(img.flatten(),pred.flatten())[0]
                    cor_lis.append(cor)
                cor = stat.pearsonr(gt.flatten(),pred.flatten())[0]
                maxpos = cor_lis.index(max(cor_lis))
        #         print(len(cor_lis),maxpos,stat.pearsonr(pred.flatten(),pred.flatten()))
                if maxpos==len(cor_lis)-1: #gt is at last pos in list
                    correct += 1
                else:
                    wrong += 1


        accuracy = correct/(correct+wrong)
        print("{}Way identification accuracy {}".format(nWay, accuracy))

    calc_identification_accuracy(nWay=2)     

    calc_identification_accuracy(nWay=5)     

    calc_identification_accuracy(nWay=10) 

preds = decoder_model.predict(Y_test_avg)
calc_metrics(preds, X_test_avg)  
