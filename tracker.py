import os
import time
import pandas as pd
import numpy as np
from numpy.random import seed
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow import set_random_seed
from skimage.transform import resize
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
seed(1)
set_random_seed(1)
np.random.seed(1)


class CentroidTracker:
    def __init__(self, detect_map):
        return
    
    def register(self, x, y):
        centroid_id = '{}'
        self.active_detect_list.append(centroid_id)
        self.detect_history.append(centroid_id)
        return
    
    def unregister(self, centroid_id):
        self.active_detect_list.pop(centroid_id)
        return