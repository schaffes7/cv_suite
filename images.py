import os
import numpy as np
from PIL import Image
from config import INPUT_HEIGHT, INPUT_WIDTH


def load_transform_image(path, final_size = (INPUT_HEIGHT, INPUT_WIDTH)):
    img = Image.open(path)
    img = img.resize(final_size)
    img = np.array(img)
    img = img.reshape((1,final_size[0],final_size[1],3))
    return img


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def sobel_filters(img):
    from scipy import ndimage
    if len(np.shape(img)) > 2:
        img = rgb2gray(img[:,:,:3])
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)


def Edger(img):
    img = sobel_filters(img)[0]
    return img


def Mask(path, img_shape = (480,640,3)):
    ext = path.split('.')[-1].lower()
    if ext == 'png':
        path = '{}\\label.png'.format(os.path.dirname(path))
    h = img_shape[0]
    w = img_shape[1]
    ar = h / w
    # READ MASK IMAGE
    img = Image.open(path)
    S = img.size
    H = S[1]; W = S[0]; AR = H / W
    # ASPECT RATIOS
    if ar != AR:
        window_s = [0,0]
        window_s[np.argmin(S)] = min(H,W)
        window_s[window_s.index(0)] = int(ar * min(S))
        top = int((H-window_s[0])/2)
        left = int((W-window_s[1])/2)
        right = W - left
        bottom = H - top
        img = img.crop((left, top, right, bottom))
    # RESIZE IMAGE (if needed)
    if H > h: img = img.resize((w,h))
    return img
