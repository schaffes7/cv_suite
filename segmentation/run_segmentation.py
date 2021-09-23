# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:08:23 2020

@author: Not Your Computer
"""
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import Image, ImageOps, ImageEnhance
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import concatenate, Dense, Activation, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, ZeroPadding2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import train_test_split
import imageio


#%%

def RGB2Gray(img):
    shape = np.shape(img)
    if len(shape) == 2: return img
    else: return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def RGB2BW(img, thresh = 100):
    shape = np.shape(img)
    if len(shape) == 2:
        return img
    else:
        fn = lambda x : 255 if x > thresh else 0
        img = PIL.Image.fromarray(img)
        return img.convert('L').point(fn, mode='1')

def Edger(img):
    img = sobel_filters(img)[0]
    return img

class Annotations(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, params, img_paths, mask_paths):
        self.batch_size = params['batch_size']
        self.img_size = params['img_shape']
        self.keep_labels = params['keep_labels']
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.color_dict = params['color_dict']
        self.classes = params['classes']
        self.n_classes = params['n_classes']
        self.n_lyrs = np.array([params['add_red'], params['add_green'], params['add_blue'], params['add_gray'], params['add_edges'], params['add_bw']]).astype(int).sum()
        self.params = params
        
    def __len__(self):
        return len(self.mask_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_img_paths = self.img_paths[i : i + self.batch_size]
        batch_mask_paths = self.mask_paths[i : i + self.batch_size]
        
        img = load_img(batch_img_paths[0], target_size = self.img_size)
        X = np.reshape(TransformImage(np.array(img), self.params), (1,480,640,self.n_lyrs))
#        X = np.zeros((1,) + self.img_size, dtype = 'float32')
#        X[0,:,:,:3] = img
#        
#        if params['add_gray']: X[0,:,:,3] = ImageOps.grayscale(img)
#        if params['add_edges']: X[0,:,:,-1] = Edger(np.array(img))
        
        y = np.zeros([self.batch_size, self.img_size[0], self.img_size[1], self.n_classes], dtype = "uint8")
        y = np.reshape(LoadMask(batch_mask_paths[0], params), (1, self.img_size[0], self.img_size[1], self.n_classes))
        
        return X, y

def RecSearch(search_dir, fname):
    results = []
    flist = os.listdir(search_dir)
    for f in flist:
        fpath = '{}\\{}'.format(search_dir, f)
        if os.path.isdir(fpath):
            results += RecSearch(fpath, fname)
        else:
            if f == fname:
                results.append(fpath)
    return results

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

def Labels(path):
    ext = path.split('.')[-1].lower()
    if ext == 'png':
        label_path = '{}\\label_names.txt'.format(os.path.dirname(path))
    else:
        label_path = path
    with open(label_path, 'r') as f:
        labels = f.read()
    labels = labels.split('\n')[:-1]
    for i in range(len(labels)):
        labels[i] = labels[i].lower()
    return labels

def Classes(img_paths):
    all_classes = []
    for path in img_paths:
        fpath = '{}\\label_names.txt'.format(os.path.dirname(path))
        with open(fpath, 'r') as f:
            classes = f.read()
        classes = classes.split('\n')[:-1]
        for i in range(len(classes)):
            classes[i] = classes[i].lower()
        all_classes += classes
    all_classes = list(set(all_classes))
    all_classes.sort()
    return all_classes

def LoadMask(path, params):
    # READ MASK IMAGE
    mask = Mask(path, img_shape = params['img_shape'])
    # READ LABELS
    labels = Labels(path)
    # CREATE OUTPUT IMAGE
    mask = np.asarray(mask)
    img = np.zeros(params['img_shape'][:2])
    for i in range(1,len(labels)):
        lab = labels[i]
        if lab in params['keep_labels']:
            lidx = params['classes'].index(lab)
            img[np.where(mask == i)] = lidx
            img = img.astype('uint8')
        else:
            img[np.where(mask == i)] = 0
            img = img.astype('uint8')
    # CONVERT TO CATEGORICAL
    img = to_categorical(img, num_classes = params['n_classes'])
    i = 0
    for lab in params['classes']:
        img[:,:,i] *= params['weight_dict'][lab]
        i += 1
    return img

def PixelDistribution(img_paths, params, logarithmic = True):
    label_counts = {}
    for path in img_paths:
        mask = Mask(path, img_size = params['img_shape'])
        labs = Labels(path)
        for lab in labs:
            cnt = len(np.where(np.asarray(mask) == labs.index(lab))[0])
            if lab not in params['classes']: lab = '_background_'
            if lab in label_counts.keys(): label_counts[lab] += cnt
            else: label_counts[lab] = cnt
    if logarithmic:
        for lab in list(label_counts.keys()):
            label_counts[lab] = np.round(np.log(label_counts[lab]),2)
    plt.figure(figsize = (10,6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.show()
    return label_counts

def CalculateWeights(label_counts):
    min_cnt = np.min(list(label_counts.values()))
    weights = {}
    for lab in list(label_counts.keys()):
        weights[lab] = 1/(label_counts[lab] / min_cnt)
    return weights

def CustomUNet(img_shape, keep_labels, net_layers = [32,64,128,256], act = 'relu', pool_size = (2,2), final_pool = (1,1), dropout = 0.50, final_act = 'softmax'):
    # INPUT
    img_input = Input(shape = img_shape)
    # ENCODING LAYERS
    fwd_lyrs = []
    i = 0
    for lyrs in net_layers:
        print('Encode: ', lyrs)
        if i == 0: x = Conv2D(lyrs, (3,3), activation = act, padding = 'same')(img_input)
        else: x = Conv2D(lyrs, (3,3), activation = act, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        print('Conv: ', list(x.get_shape()))
        fwd_lyrs.append(x)
        x = AveragePooling2D(pool_size)(x)
        print('Pool: ', list(x.get_shape()))
        i += 1
    # MIDDLE LAYER
    print('Switch-board: ', len(keep_labels)+1)
    act = 'relu'
    x = Conv2D(len(keep_labels)+1, (3,3), activation = act, padding = 'same')(x)
    print('Conv: ', list(x.get_shape()))
    net_layers.reverse()
    fwd_lyrs.reverse()
    # DECODING LAYERS
    i = 0
    for lyrs in net_layers:
        print('Decode: ', lyrs)
        x = Conv2D(lyrs, (3,3), activation = act, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        print('Conv: ', list(x.get_shape()))
        x = concatenate([UpSampling2D(pool_size)(x), fwd_lyrs[i]])
        print('Up2D: ', list(x.get_shape()))
        i += 1
    # OUTPUT
    out = Conv2D(len(keep_labels)+1, final_pool, activation = final_act, padding = 'same')(x)
    print('Conv: ', list(out.get_shape()))
    model = Model(img_input, out)
    return model

def Colors(path, label = None):
    img = plt.imread(path)
    img *= 255
    img = img.astype(int)
    if label == None:
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        colors = np.append(colors, np.zeros((len(colors), 1), dtype = int), axis = 1)
        for i in range(len(colors)):
            colors[i,3] = len(np.where(img==colors[i,0:3])[0])
    else:
        mask = Mask(path)
        labels = Labels(path)
        if label in labels:
            lidx = labels.index(label)
            img[np.where(mask != lidx)] = 0
            colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
            colors = np.append(colors, np.zeros((len(colors), 1), dtype = int), axis = 1)
            for i in range(len(colors)):
                colors[i,3] = len(np.where(img==colors[i,0:3])[0])
        else:
            return np.array([])
    return colors

def ColorSpace(path, cname):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import colors
    labels = Labels(path)
    mask = np.asarray(Mask(path))
    cidx = labels.index(cname)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[np.where(mask != cidx)] = 0
    r, g, b = cv2.split(img)
    fig = plt.figure(figsize = (8,8))
    axis = fig.add_subplot(1, 1, 1, projection = "3d")
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin = -1., vmax = 1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors = pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

def sobel_filters(img):
    from scipy import ndimage
    if len(np.shape(img)) > 2:
        img = RGB2Gray(img[:,:,:3])
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def TransformImage(img, params):
    # Original RGB Layers
    R = params['add_red']; G = params['add_green']; B = params['add_blue']
    # Custom Layers
    Gray = params['add_gray']; BW = params['add_bw']; Edges = params['add_edges']
    rgb = [R,G,B]
    n_colors = np.array(rgb).astype(int).sum()
    lyrs = [R,G,B,Gray,BW,Edges]
    n_lyrs = np.array(lyrs).astype(int).sum()
    if n_lyrs == 0: return img
    # INITIALIZE ARRAY
    if n_colors > 0: out = img[:,:,rgb]
    # ADD CUSTOM LAYERS
    if Gray: out = np.append(out, np.reshape(RGB2Gray(img), (480,640,1)), axis = 2)
    if Edges: out = np.append(out, np.reshape(Edger(img), (480,640,1)), axis = 2)
    if BW: out = np.append(out, np.reshape(RGB2BW(img), (480,640,1)), axis = 2)
    return out

def FindCentroids(layer, method = 'mean', buffer = 15):
    centroids = {}
    uniques = np.unique(layer)
    if np.sum(layer) <= 0: centroids[0] = (0,0)
    if np.sum(layer) > 0:
        for u in uniques:
            if u >= 0:
                coords = np.where(layer == u)
                if method == 'mean':
                    row = int(np.mean(coords[1][buffer:len(coords[1])-buffer]))
                    col = int(np.mean(coords[0][buffer:len(coords[0])-buffer]))
                if method == 'median':
                    row = int(np.median(coords[1][buffer:len(coords[1])-buffer]))
                    col = int(np.median(coords[0][buffer:len(coords[0])-buffer]))
                try: spread = (int(np.std(coords[1])), int(np.std(coords[0])))     
                except: spread = (0,0)
                centroids[u] = [(row,col), spread]       
    return centroids


#%%
#==============================================================================
#                         DEFINE SCRIPT PARAMETERS
#==============================================================================

params = {'img_dir':"D:\\Images\\_Annotated",
          'model_dir':'D:\\CVFiles\\ANNIE\\Models',
          'outfile':'D:\\CVFiles\\ANNIE\\Models\\segmentor.h5',
          'add_red':True,
          'add_green':True,
          'add_blue':True,
          'add_gray':True,
          'add_bw':True,
          'add_edges':True,
          'tune_weights':False,
          'frame_h':480,
          'frame_w':640,
          'n_channels':3,
          'use_rollups':False,
          'keep_labels':['logan'],
          'rollup':{'logan':'dog',
                    'morgan':'dog',
                    'dog_bed':'furniture',
                    'table':'furniture',
                    'chair':'furniture',
                    'rug':'furniture',
                    'desk':'furniture',
                    'cabinet':'wall',
                    'door':'wall',
                    'floor':'floor',
                    'tile':'floor',
                    'wall':'wall',
                    'toy':'item',
                    'ball':'item',
                    'mouse':'item',
                    'cup':'item',
                    'keyboard':'item',
                    'phone':'item',
                    'wallet':'item',
                    'monitor':'item',
                    'computer':'item',
                    'french_press':'item',
                    'other':'item',
                    '_background_':'room'},
            'net_layers':[32,64,128,256,512],
          'n_epochs':3,
          'test_ratio':0.0,
          'batch_size':2,
          'patience':2,
          'baseline':0.0,
          'dropout':0.10,
          'loss':'mae',
          'act':'selu',
          'final_act':'softmax',
          'opt_type':'adam',
          'learn_rate':1e-5,
          'pool_size':(2,2),
          'final_pool':(1,1),
          'metric':'categorical_accuracy',
          'val_metric':'val_categorical_accuracy',
          
          'camera_input':0,
          'add_centroids':True,
          'font':cv2.FONT_HERSHEY_DUPLEX,
          'stop_camera_key':'q',
          'alpha':0.50,
          'hud_color':(220,220,50),
          'hud_alpha':0.85,
          'prediction_alpha':0.65,
          'line_type':cv2.LINE_AA}

n_lyrs = np.array([params['add_red'], params['add_green'], params['add_blue'], params['add_gray'], params['add_edges'], params['add_bw']]).astype(int).sum()
img_shape = (480,640,n_lyrs)
params['img_shape'] = img_shape

color_dict = {0:[0,0,0],
                  1:[128,0,0],2:[0,128,0],3:[128,128,0],
                  4:[0,0,128],5:[128,0,128],6:[0,128,128],
                  7:[128,128,128],8:[64,0,0],9:[192,0,0],
                  10:[64,128,0],11:[192,128,0],12:[64,0,128],
                  13:[192,0,128],14:[64,128,128],15:[192,128,128],
                  16:[0,64,0],17:[128,64,0],18:[0,192,0],
                  19:[128,192,0],20:[0,64,128],21:[128,64,128],
                  22:[0,192,128],23:[128,192,128],24:[64,64,0],
                  25:[192,64,0],26:[64,192,0]}
params['color_dict'] = color_dict
params['classes'] = ['_background_'] + params['keep_labels']
params['n_classes'] = len(params['classes'])
params['weight_dict'] = {}
for lab in params['classes']:
    params['weight_dict'][lab] = 1.0


#%%
#==============================================================================
#                              COLLECT IMAGES
#==============================================================================

# FIND RELEVANT FILEPATHS
img_paths = RecSearch(params['img_dir'], 'img.png')

# FILTER CLASSES / IMAGES
remove_img_paths = []
for path in img_paths:
    labs = Labels(path)
    keep_img = False
    for c in params['keep_labels']:
        if c in labs:
            keep_img = True
    if not keep_img:
        remove_img_paths.append(path)
for path in remove_img_paths:
    if path in img_paths:
        img_paths.remove(path)

random.shuffle(img_paths)

mask_paths = []; label_paths = []
for path in img_paths:
    label_path = '{}\\label_names.txt'.format(os.path.dirname(path))
    mask_path = '{}\\label.png'.format(os.path.dirname(path))
    label_paths.append(label_path)
    mask_paths.append(mask_path)


#%%
#==============================================================================
#                             TUNE CLASS WEIGHTS
#==============================================================================

if params['tune_weights']:
    print('\nTUNING WEIGHTS...')
    label_counts = PixelDistribution(img_paths, params, logarithmic = False)
    print(label_counts)
    params['weight_dict'] = CalculateWeights(label_counts)


#%%
#==============================================================================
#                      CREATE TRAIN & TEST GENERATORS
#==============================================================================

# TRAIN / TEST SPLITS
if params['test_ratio'] > 0:
    train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = train_test_split(img_paths, mask_paths, test_size = params['test_ratio'], shuffle = True)
else:
    train_img_paths = img_paths; train_mask_paths = mask_paths
    test_img_paths = img_paths; test_mask_paths = mask_paths

# CUSTOM TRAIN GENERATORS
train_gen = Annotations(params, train_img_paths, train_mask_paths)
val_gen = Annotations(params, test_img_paths, test_mask_paths)


#%%
#==============================================================================
#                              LEARN MACHINES
#==============================================================================

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# BUILD MODEL
print('\nBuilding Model...')
model = CustomUNet(img_shape = img_shape,
                   keep_labels = params['keep_labels'],
                   net_layers = params['net_layers'],
                   dropout = params['dropout'],
                   act = params['act'],
                   pool_size = params['pool_size'],
                   final_act = params['final_act'],
                   final_pool = params['final_pool'])

# COMPILE MODEL
print('\nCompiling Model...')
#model.compile(optimizer = params['opt_type'], loss = params['loss'], metrics = ['categorical_accuracy'])
model.compile(optimizer = params['opt_type'],
              loss = params['loss'],
              metrics = [tf.keras.metrics.MeanIoU(num_classes = params['n_classes'])])

callbacks = [keras.callbacks.ModelCheckpoint(params['outfile'], save_best_only = True),
             keras.callbacks.EarlyStopping(monitor = 'loss', verbose = 1, patience = params['patience'], baseline = params['baseline'])]

# TRAIN MODEL
histories = []
print('\n[CV_SegmentationModel]: TRAINING MODEL...')
histories.append(model.fit(train_gen,
                           epochs = params['n_epochs'],
                           validation_data = val_gen,
                           callbacks = callbacks))

# PLOT HISTORY
if params['n_epochs'] > 1 and params['patience'] > 1:
    plt.plot(histories[0].history[params['metric']])
    plt.plot(histories[0].history[params['val_metric']])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#%%
#==============================================================================
#                          PREVIEW PREDICTIONS
#==============================================================================

#frames = []
#for epoch in range(params['n_epochs']):
#    j = random.randint(0,100)
#    j = test_img_paths.index(r'D:\Images\_Annotated\196_122320_215014_348415\img.png')
#    X_test, y_test = train_gen.__getitem__(j)
#    y_test = np.argmax(y_test[0],-1)
#    for i in range(len(train_img_paths)):
#        X, y = train_gen.__getitem__(i)
#        model.fit(X, y, verbose = 0)
#        
#        true_color_map = np.zeros([480,640,3])
#        for val in np.unique(y_test):
#            true_color_map[np.where(y_test == val)] = color_dict[val]
#        true_color_map = true_color_map.astype('uint8')
#        
#        p = model.predict(X_test)
##        e = p - y[0]
##        mae = np.mean(np.abs(e))
#        p = np.argmax(p[0], axis = -1)
#        p = p.astype('uint8')
#        
#        color_map = np.zeros([480,640,3])
#        for val in np.unique(p):
#            color_map[np.where(p == val)] = color_dict[val]
#        color_map = color_map.astype('uint8')
#        im = (255*plt.imread(train_img_paths[j])).astype('uint8')
#        color_map = cv2.addWeighted(im, 0.50, color_map, 0.50, 0)
#        frames.append(color_map)
#        
#        color_y = np.zeros([480,640,3])
#        y = np.argmax(y,-1)
#        for val in np.unique(y):
#            color_y[np.where(y[0] == val)] = color_dict[val]
#        color_y = color_y.astype('uint8')
#        
#        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (9,6))
#        axes[0].imshow(color_y)
#        axes[1].imshow(color_map)
#        axes[2].imshow(true_color_map)
#        fig.tight_layout()
#        plt.show()

#%%
# SAVE TEST IMAGE HISTORY TO VIDEO
#total_frames = len(frames)
#color_percentage_for_each_frame = (100 / total_frames) / 100
#write_to = 'D:\\Images\\training.mp4'
#writer = imageio.get_writer(write_to, format = 'mp4', mode = 'I', fps = 10)
#for frame in frames:
#        writer.append_data(frame)
#writer.close()


#%%
#==============================================================================
#                              RUN CAMERA
#==============================================================================

# CREATE VIDEO CAPTURE OBJECT
cap = cv2.VideoCapture(params['camera_input'])
frame_no = 0
p_thresh = 0.65
n_classes = params['n_classes']
img_shape = params['img_shape']
min_frame_percentage = 0.01

while(True):
    # CAPTURE FRAME
    ret, frame = cap.read()
    output = frame.copy()
    inframe = TransformImage(frame, params)
    p = model.predict(np.array([inframe]))[0]
    p[np.where(p < p_thresh)] = 0
#    if n_classes > 2: p = np.argmax(p[:,:,1:], axis = -1) + 1
#    else: p = np.argmax(p, axis = -1)
    p = np.argmax(p, axis = -1)
    p = p.astype('uint8')
    
    # CREATE COLORMAP OF PREDICTIONS
    color_map = np.zeros([480,640,3])
    for val in np.unique(p):
        color_map[np.where(p == val)] = color_dict[val]
    color_map = color_map.astype('uint8')
    color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    
    # OVERLAY PREDICTIONS
    output = cv2.addWeighted(color_map, params['alpha'], frame, 1-params['alpha'], 0)
    
    # FIND CENTROIDS FOR EACH LABEL
    if params['add_centroids']:
        try:
            centroids = FindCentroids(p, method = 'median')
            for val in np.unique(p):
                if (p == val).astype(int).sum() / 307200 > min_frame_percentage:
                    label = params['classes'][val]
                    if params['rollup'][label] in ['item','dog']:
                        [position, spread] = centroids[val]
                        # DRAW BOUNDARIES ON FRAME
#                        start_pt = (int(R-spread/2), int(C-spread/2))
#                        end_pt = (int(R+spread,/2), int(C+spread/2))
                        cv2.rectangle(output, tuple((np.array(position)-np.array(spread)).astype(int)), tuple((np.array(position)+np.array(spread)).astype(int)), 255, 2)
#                        cv2.ellipse(output,position,spread,0,0,360,255,2)
#                        cv2.circle(output, position, int(np.mean(spread)),  color_dict[val], 2)
        except:
            pass
    # DISPLAY FRAME
    cv2.imshow('frame', output)

    if cv2.waitKey(1) & 0xFF == ord(params['stop_camera_key']):
        break
    
    frame_no += 1

# DISCONNECT CAMERA FEED
cap.release()
cv2.destroyAllWindows()