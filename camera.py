import os
import time
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import imagenet_utils, MobileNet
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFilter, ImageEnhance
from utils import generate_anchors
from config import (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS,
                    INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
                    DETECTOR_PATH, CLASSIFIER_PATH, 
                    FPS_SMOOTHING, HUD_COLOR, HUD_ALPHA)


# Generate Frame Anchors
anchors = generate_anchors()

# READ MODEL PARAMETERS FILE
model_params_path = "D:\\CVFiles\\ANNIE\\Models\\classes.txt".format(os.path.basename(params['classifier_path']))
with open(model_params_path, 'r') as f:
    classes = f.read()
classes = classes.split(',')

# Import Models
detector = load_model(DETECTOR_PATH)
classifier = load_model(CLASSIFIER_PATH)

histories = {}
C_histories = {}
R_histories = {}
for i in range(len(classes)):
    histories[i] = list(np.zeros(FPS_SMOOTHING))
    R_histories[i] = list(np.zeros(FPS_SMOOTHING))
    C_histories[i] = list(np.zeros(FPS_SMOOTHING))

# Create Video Capture Object
cap = cv2.VideoCapture(0)
st = time.time()
frame_no = 0
fps = 0


while(True):
    
    # Capture frame
    ret, frame = cap.read()
    output = frame.copy()
    frame = np.array(frame)
    
    # SEGMENT FRAME
    layers = DET.predict(np.reshape(frame, (1, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)))
    L = 0
    n_layers = 1
    for lidx in range(n_layers):
        lyr = layers[lidx]
        
        # DENOISE LAYER
#        lyr = cv2.fastNlMeansDenoising(layers[lidx], h = 3)
        
        # MAKE DETECTIONS (currently assumes 1 at most per layer)
        [centroid, spread] = Detector(lyr, method = 'mean', thresh = DETECT_THRESH)
        layer_color = class_colors[L]
        
        hot_idx = np.where(lyr > 0)
        lyr_out = np.zeros([FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS], dtype = np.uint8)
        lyr_out[hot_idx[0], hot_idx[1], 0] = layer_color[0]
        lyr_out[hot_idx[0], hot_idx[1], 1] = layer_color[1]
        lyr_out[hot_idx[0], hot_idx[1], 2] = layer_color[2]
        
        # ADD SEGMENTATION OVERLAY
        output = cv2.add(lyr_out, output)

        # IF NO DETECTS, SKIP TO NEXT FRAME
        if centroid == [0,0] and spread == 0:
            pass
        
        # ELSE CLASSIFY DETECTS
        else:
            # GENERATE SUBFRAME
            R = centroid[0]; C = centroid[1]
            x1 = C - spread; x2 = C + spread
            y1 = R - spread; y2 = R + spread
            sf = frame[y1:y2, x1:x2, :]
            sf = np.resize(np.array(sf), (1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
            
            # CLASSIFY DETECTS
            pc = CLS.predict(sf)
            cidx = np.argmax(pc)
            cname = classes[cidx]
            c_color = class_colors[cidx]
            
            # DRAW CLASS DETAILS
            draw_class_details(output, params, class_text = True, bar_chart = False)
            
            # DRAW CIRCLE ON FRAME
            rad_list = histories[L]
            rad_list = rad_list[1:] + [spread]
            rad = int(round(np.mean(rad_list)))
            cv2.circle(output, (C,R), rad,  c_color, 2)
            
            # UPDATE HISTORIES
            histories[L] = rad_list
            
        L += 1
    
    # CALCULATE FPS
    if frame_no % FPS_SMOOTHING == 0:
        FPS = round(FPS_SMOOTHING / (time.time() - st))
        st = time.time()
        
    # DRAW HUD OVERLAY
    draw_hud(output)
    
    # SHOW FRAME
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord(STOP_CAMERA_KEY):
        break
    
    frame_no += 1

# DISCONNECT CAMERA FEED
cap.release()
print('\nCAMERA CONNECTION RELEASED.')
cv2.destroyAllWindows()

print('\nDONE.')