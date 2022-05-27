
import copy
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import backend as K
import sys
from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from Models.encdec_model import *
from utils.data_loader import *
from utils.misc import *
from utils.loss_utils import *
from encdec_train import train_encoder_decoder_and_rev
from encoder_train import train_encoder
from Models.encoder import *
from Models.decoder import *
from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import configs.config as config
from test import testing


if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    # parser = ArgumentParser()
    # parser.add_argument("--mode",default="train", choices=["train","test"],help="train/test")
    # parser = ArgumentParser()
    # parser.set_defaults(verbose=False)
    # opt = parser.parse_args()
    mode = 'test'
    if mode == 'train':
        #Train encoder
        if (os.path.exists(config.encoder_weights) and not config.retrain_encoder):
            
            print('pretrained encoder weights file exist')
            handler = data_handler(matlab_file = config.kamitani_data_mat)
            Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0)
            NUM_VOXELS = Y.shape[1]
            encoder_model = encoder(NUM_VOXELS)
            encoder_model.trainable = False
            print(config.encoder_weights)
            encoder_model.load_weights(config.encoder_weights )
        else:
            print('training encoder')  
            encoder_model = train_encoder(config)  #img -> fMRI
            encoder_model.trainable = False #encoder is freezed

        # train decoder
        print("training decoder")
        train_encoder_decoder_and_rev(config,encoder_model) 
    else:
        print(config.kamitani_data_mat)
        testing(config)



