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
                    FPS_SMOOTHING, HUD_COLOR, HUD_ALPHA, HUD_FONT)
from video import hud


# Generate Frame Anchors
anchors = generate_anchors()

# Import Models
detector = load_model(DETECTOR_PATH)
classifier = load_model(CLASSIFIER_PATH)

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
    
    # For each anchor...
    for anchor in anchors:
        # Run detection model
        detection = detector.predict()
        if detection >= DETECT_THRESH:
            # Run classification model
            predicted_class = classifier.predict()
    
    # Draw overlays
    hud(output)
    
    # Render frame
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord(STOP_CAMERA_KEY):
        break
    
    frame_no += 1

# Disconnect camera feed
cap.release()
print('\nCAMERA CONNECTION RELEASED.')
cv2.destroyAllWindows()