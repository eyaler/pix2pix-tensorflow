"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import os

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, load_size=286, fine_size=256, aspect=False, pad_to_white=0, which_direction='AtoB', gcn=False, interp=True, flip=True, rot=False, is_test=False, is_grayscale_A=False, is_grayscale_B=False):
    img_A, img_B = load_image(image_path, is_grayscale_A=is_grayscale_A, is_grayscale_B=is_grayscale_B)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, load_size=load_size, fine_size=fine_size, aspect=aspect, pad_to_white=pad_to_white, which_direction=which_direction, gcn=gcn, interp=interp, flip=flip, rot=rot, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    if is_grayscale_A:
        img_A = img_A[..., None]

    if is_grayscale_B:
        img_B = img_B[..., None]

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def rgb2gray(rgb):
    return rgb.dot([0.299, 0.587, 0.114])

def load_image(image_path, is_grayscale_A, is_grayscale_B):
    input_img = imread(image_path)
    w = input_img.shape[1]//2
    img_A = input_img[:, :w]
    img_B = input_img[:, w:]

    if is_grayscale_A:
        img_A = rgb2gray(img_A)

    if is_grayscale_B:
        img_B = rgb2gray(img_B)

    return img_A, img_B

def myresize(img, dims, aspect, pad_to_white, interp):
    if interp:
        interp = 'bilinear'
    else:
        interp = 'nearest'
    if aspect:
        diff = img.shape[0]-img.shape[1]
        padding = (abs(diff) // 2, abs(diff) - abs(diff) // 2)
        if diff < 0:
            vals = [padding, (0, 0)]
        else:
            vals = [(0, 0), padding]
        if len(img.shape)==3:
            vals.append((0,0))
        img = np.pad(img, vals, 'constant', constant_values=255 if pad_to_white else 0)
    return scipy.misc.imresize(img, dims, interp=interp) # accepts (x,y) or (x,y,3) image sizes

def gcn_norm(img):
    minimum = np.min(img, axis=(0,1), keepdims=True)
    maximum = np.max(img, axis=(0,1), keepdims=True)
    return 255*(img-minimum)/(maximum-minimum)

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, aspect=False, pad_to_white=False, which_direction='AtoB', gcn=False, interp=True, flip=True, rot=False, is_test=False):

    if gcn:
        if which_direction=='AtoB':
            img_A = gcn_norm(img_A)
        else:
            img_B = gcn_norm(img_B)

    interp_A = True
    interp_B = True
    if not interp:
        if which_direction=='AtoB':
            interp_B = False
        else:
            interp_A = False

    if is_test:
        img_A = myresize(img_A, [fine_size, fine_size], aspect, pad_to_white, interp_A)
        img_B = myresize(img_B, [fine_size, fine_size], aspect, pad_to_white, interp_B)
    else:
        img_A = myresize(img_A, [load_size, load_size], aspect, pad_to_white, interp_A)
        img_B = myresize(img_B, [load_size, load_size], aspect, pad_to_white, interp_B)

        h1 = np.random.randint(load_size-fine_size+1)
        w1 = np.random.randint(load_size-fine_size+1)
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

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

