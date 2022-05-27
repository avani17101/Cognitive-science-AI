
import os


GPU_ID = "0"
imagenet_dir = "/ssd_scratch/cvit/bvk_cvit/image_net/"
imagenet_wind_dir = imagenet_dir
external_images_dir =  os.path.join(imagenet_dir,"/ssd_scratch/cvit/bvk_cvit/ImageNet-2015/ILSVRC2015/Data/VID/val/")

project_dir = "/home/bvk_cvit/ssfmri2im/"
images_npz = os.path.join(project_dir,"data/images.npz")
kamitani_data_format = True
kamitani_data_mat = os.path.join(project_dir,"data/Subject3.mat")
caffenet_models_weights = os.path.join(project_dir,"models/imagenet-caffe-ref.mat")
results  = os.path.join(project_dir,"results/")


encoder_weights = os.path.join(project_dir,"models/encoder.hdf5")
retrain_encoder = False
decoder_weights = os.path.join('models/',"decoder.hdf5")


DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat'
FILENAME = 'imagenet-caffe-ref.mat'
EXPECTED_BYTES = 228031200


image_size = 112
encoder_epochs = 2
decoder_epochs = 2

w1 = 0.15
w2 = 0.7
w3 = 0.15

w_mae_vox = 1
w_cs_vox = 0.1