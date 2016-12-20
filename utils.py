"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, load_size=286, fine_size=256, aspect=False, flip=True, rot=False, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, load_size=load_size, fine_size=fine_size, aspect=aspect, flip=flip, rot=rot, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size or fine_size*input_aspect_ratio, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, aspect=False, flip=True, rot=False, is_test=False):
    ratio = 1
    if aspect:
        ratio = float(img_A.shape[0])/img_A.shape[1]
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, int(fine_size*ratio)])
        img_B = scipy.misc.imresize(img_B, [fine_size, int(fine_size*ratio)])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, int(load_size*ratio)])
        img_B = scipy.misc.imresize(img_B, [load_size, int(load_size*ratio)])

        h1 = int(np.ceil(np.random.uniform(1e-2, int((load_size-fine_size)*ratio))))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+int(fine_size*ratio), w1:w1+fine_size]
        img_B = img_B[h1:h1+int(fine_size*ratio), w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

        if rot:
            r = np.random.choice(4)
            img_A = np.rot90(img_A, r)
            img_B = np.rot90(img_B, r)


    return img_A, img_B

# -----------------------------

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.


