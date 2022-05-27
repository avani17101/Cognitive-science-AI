
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from Models.encdec_model import *
from Models.encoder import *
from Models.decoder import *
from utils.data_loader_utils import *
from utils.misc_utils import *
from utils.loss_utils import *
import configs.config as config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_encoder(config_file):
    '''
    encoder is trained in supervised manner entirely
    '''
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    data_loader = data_handler(matlab_file = config_file.kamitani_data_mat)

    sess = tf.compat.v1.Session(config=gpu_config)
    tf.compat.v1.keras.backend.set_session(sess)


    Y,Y_test,Y_test_avg = data_loader.get_data(roi = 'ROI_VC')
    labels_train, labels = data_loader.get_labels()
    NUM_VOXELS = Y.shape[1]

    data_imgs= np.load(config_file.images_npz)
    X, X_test = data_imgs['train_images'][labels_train],data_imgs['test_images']
    X_test_sorted = X_test
    X_test = X_test[labels]

    initial_lrate = 0.1

    if (not os.path.exists(config_file.caffenet_models_weights)):
        download_network_weights()
   
        
    enc_model = encoder(NUM_VOXELS)
    enc_model.compile(loss=combined_loss, optimizer= SGD(lr=initial_lrate,decay = 0.0 , momentum = 0.9,nesterov=True),metrics=['mse','cosine_similarity','mae'])
    print(enc_model.summary())

    callbacks = []

    reduce_lr = LearningRateScheduler(step_decay)
    callbacks.append(reduce_lr)


    train_generator = encoder_dataloader(X, Y, batch_size=64,max_shift = 5)
    test_generator = encoder_dataloader(X_test_sorted, Y_test_avg, batch_size=50,max_shift = 0)

    enc_model.fit_generator(train_generator, epochs=config_file.encoder_epochs,validation_data=test_generator ,verbose=2,use_multiprocessing=False,callbacks=callbacks) #, steps_per_epoch=1200//64 , validation_steps=1
    if(config_file.encoder_weights is not None):
        enc_model.save_weights(config_file.encoder_weights)

    return enc_model