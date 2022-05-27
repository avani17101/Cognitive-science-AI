import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import configs.config as config
import os
import cv2


# Create Image dataset from Imagenet folders
def image_generate(imgnet_dir = config.imagenet_wind_dir,test_csv='./imageID_test.csv',train_csv='./imageID_training.csv',size = config.image_size,out_file= config.images_npz,interpolation = 'cubic'):
    test_im = pd.read_csv(test_csv,header=None)
    train_im = pd.read_csv(train_csv,header=None)

    test_images = np.zeros([50, size, size, 3])
    train_images = np.zeros([1200, size, size, 3])

    count = 0

    for file in list(test_im[1]):
        # folder = file.split('_')[0]
        img = plt.imread(imgnet_dir + '/test/' + file)
        test_images[count] = transform_img(img, size,interpolation)
        count += 1

    count = 0

    for file in list(train_im[1]):
        # folder = file.split('_')[0]
        img = plt.imread(imgnet_dir + '/training/' + file)
        train_images[count] = transform_img(img, size,interpolation)
        count += 1
    np.savez(out_file, train_images=train_images, test_images=test_images)


#ceneter crop and resize
def image_prepare(img,size,interpolation):

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

if __name__ == "__main__":
    if(os.path.exists(config.images_npz)):
        print('images npz file exists')
    else:
        print('creating npz file exists')
        image_generate()