import numpy as np
import scipy.stats as stat
import os
from six.moves import urllib
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras as keras
import configs.config as config

from utils.loss_utils import *
import configs.config as config

from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from utils.loss_utils import *
from utils.data_loader import *
from Models.encdec_model import *
from scipy.ndimage import shift

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_imgs_in_collage(img_arrays, rows =10, border =5,save_file = None):
    """
    create image collage for arrays of images
    """

    img_len =img_arrays[0].shape[2]
    array_len  = img_arrays[0].shape[0]
    num_arrays =len(img_arrays)

    cols = int(np.ceil(array_len/rows))
    img_collage = np.ones([rows * (img_len + border) + border,num_arrays *cols * (img_len + border) , 3])

    for ind in range(array_len):
        x = (ind % cols) * num_arrays
        y = int(ind / cols)

        img_collage[border * (y + 1) + y * img_len:border * (y + 1) + (y + 1) * img_len, cols * (x + 1) + x * img_len:cols * (x + 1) +(x + num_arrays) * img_len]\
            = np.concatenate([img_arrays[i][ind] for i in range(num_arrays) ],axis=1)

    if(save_file is not None):
        plt.imsave(save_file,img_collage)

    return img_collage



def save_images(images,images_orig = None ,folder=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(images.shape[0]):
        if(images_orig is None):
            plt.imsave(folder+'img_'+str(i)+'.jpg',images[i])
        else:
            img_concat = np.concatenate([images_orig[i],images[i]],axis=1)
            img_concat = np.squeeze(img_concat)
            plt.imsave(folder + 'img_' + str(i) + '.jpg', img_concat)

def transform_img(img,size,interpolation = 'cubic'):
    """
    Select central crop, resize and convert gray to 3 channel image

    """
    out_img = np.zeros([size,size,3])
    s = img.shape
    r = s[0]
    c = s[1]

    trimSize = np.min([r, c])
    lr = int((c - trimSize) / 2)
    ud = int((r - trimSize) / 2)
    img = img[ud:min([(trimSize + 1), r - ud]) + ud, lr:min([(trimSize + 1), c - lr]) + lr]

    img = cv2.resize(img, (size, size))
    if (np.ndim(img) == 3):
        out_img = img
    else:
        out_img[ :, :, 0] = img
        out_img[ :, :, 1] = img
        out_img[ :, :, 2] = img

    return out_img/255.0



def shift_img_randomly(img,max_shift = 0 ):
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted


def snr_calc(y, y_avg, labels):
    sig = np.var(y_avg, axis=0)
    noise = 0
    for l in labels:
        noise += np.var(y[labels == l], axis=0)
    noise /= len(labels)
    return sig/noise


def corr_percintiles(y,y_pred, per = [50,75,90]):
    num_voxels = y.shape[1]
    corr = np.zeros([num_voxels])

    for i in range(num_voxels):
        corr[i] = stat.pearsonr(y[:, i], y_pred[:, i])[0]
    corr = np.nan_to_num(corr)

    corr_per = []
    for p in per:
        corr_per.append(np.percentile(corr,p))
    return corr_per

class log_save_imgs_in_collage_callback(keras.callbacks.Callback):
    def __init__(self, Y, X, model, dir = '',freq = 10):
        self.Y = Y
        self.X = X
        self.pred_model = model
        self.freq = freq
        self.dir = dir

    def on_epoch_end(self, epoch, logs={}):
        if(epoch%self.freq==0):
            X_pred = self.pred_model.predict(self.Y)
            collage = save_imgs_in_collage([self.X,X_pred])
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)


            plt.imsave(self.dir+'ep_'+str(epoch)+'.jpg',collage)



def download_network_weights(download_link = config.DOWNLOAD_LINK, file_name = config.caffenet_models_weights, expected_bytes = config.EXPECTED_BYTES):
    """ Download the pretrained model if it's not already downloaded """

    print("Downloading the pre-trained model.")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded pre-trained model', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')


