#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np

import cv2

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)

    im[:, :, 0] = ((im[:, :, 0] + hue * 255) % 255).astype(np.uint8)
    im[:, :, 1] = np.clip(im[:, :, 1] * sat, 0, 255).astype(np.uint8)
    im[:, :, 2] = np.clip(im[:, :, 2] * val, 0, 255).astype(np.uint8)

    return cv2.cvtColor(im, cv2.COLOR_HSV2RGB_FULL)

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res  = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.shape[0]
    ow = img.shape[1]
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    pad = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT)
    cropped = pad[dh+ptop:dh+ptop + sheight - 1,dw+pleft:dw+pleft + swidth - 1,:]

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cv2.resize(np.array(cropped), shape, interpolation=cv2.INTER_LINEAR)

    if flip: 
        sized = cv2.flip(sized, 1)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,21))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 21))
        cc = 0
        for i in range(bs.shape[0]):
            x0 = bs[i][1]
            y0 = bs[i][2]
            x1 = bs[i][3]
            y1 = bs[i][4]
            x2 = bs[i][5]
            y2 = bs[i][6]
            x3 = bs[i][7]
            y3 = bs[i][8]
            x4 = bs[i][9]
            y4 = bs[i][10]
            x5 = bs[i][11]
            y5 = bs[i][12]
            x6 = bs[i][13]
            y6 = bs[i][14]
            x7 = bs[i][15]
            y7 = bs[i][16]
            x8 = bs[i][17]
            y8 = bs[i][18]

            x0 = min(0.999, max(0, x0 * sx - dx)) 
            y0 = min(0.999, max(0, y0 * sy - dy)) 
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            x3 = min(0.999, max(0, x3 * sx - dx))
            y3 = min(0.999, max(0, y3 * sy - dy))
            x4 = min(0.999, max(0, x4 * sx - dx))
            y4 = min(0.999, max(0, y4 * sy - dy))
            x5 = min(0.999, max(0, x5 * sx - dx))
            y5 = min(0.999, max(0, y5 * sy - dy))
            x6 = min(0.999, max(0, x6 * sx - dx))
            y6 = min(0.999, max(0, y6 * sy - dy))
            x7 = min(0.999, max(0, x7 * sx - dx))
            y7 = min(0.999, max(0, y7 * sy - dy))
            x8 = min(0.999, max(0, x8 * sx - dx))
            y8 = min(0.999, max(0, y8 * sy - dy))
            
            bs[i][1] = x0
            bs[i][2] = y0
            bs[i][3] = x1
            bs[i][4] = y1
            bs[i][5] = x2
            bs[i][6] = y2
            bs[i][7] = x3
            bs[i][8] = y3
            bs[i][9] = x4
            bs[i][10] = y4
            bs[i][11] = x5
            bs[i][12] = y5
            bs[i][13] = x6
            bs[i][14] = y6
            bs[i][15] = x7
            bs[i][16] = y7
            bs[i][17] = x8
            bs[i][18] = y8
            
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def change_background(img, mask, bg):
    return cv2.copyTo(img, mask, cv2.resize(bg, (img.shape[1], img.shape[0])))

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, bgpath):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    maskpath = imgpath.replace('JPEGImages', 'mask').replace('/00', '/').replace('.jpg', '.png')

    ## data augmentation
    img = cv2.imread(imgpath)
    mask = cv2.imread(maskpath)
    bg = cv2.imread(bgpath)

    img = change_background(img, mask, bg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    ow, oh = img.shape[1], img.shape[0]
    label = fill_truth_detection(labpath, ow, oh, flip, dx, dy, 1./sx, 1./sy)
    return img,label

