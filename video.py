import pandas as pd
import numpy as np
import random
import cv2
from config import (FRAME_HEIGHT, FRAME_WIDTH, INPUT_HEIGHT, INPUT_WIDTH,
                    SEARCH_SCALES, DETECT_THRESH, HUD_ALPHA)


def denoise(lyr):
    return cv2.fastNlMeansDenoising(lyr, h = 3)


def Intersection(boxA, boxB):
    # CALCULATE THE OVERLAPPING AREA OF TWO RECTANGLES
    # boxA and boxB are arrays of [x1,y1,x2,y2], where (x2,y2) is the lower right corner
    x1, y1 = (max(boxA[0],boxB[0]), max(boxA[1],boxB[1]))
    x2, y2 = (min(boxA[2],boxB[2]), min(boxA[3],boxB[3]))
    if x2-x1 > 0 and y2-y1 > 0:
        return (x2-x1) * (y2-y1)
    else:
        return 0


def Union(boxA, boxB):
    # CALCULATE THE UNIONED AREA OF TWO RECTANGLES
    # boxA and boxB are arrays of [x1,y1,x2,y2], where (x2,y2) is the lower right corner
    boxA_area = abs((boxA[0]-boxA[2])*(boxA[1]-boxA[3]))
    boxB_area = abs((boxB[0]-boxB[2])*(boxB[1]-boxB[3]))
    if Intersection(boxA, boxB) > 0:
        U = boxA_area + boxB_area - Intersection(boxA, boxB)
        return U
    else:
        return 0


def IOU(boxA, boxB):
    # CALCULATE THE INTERSECTION OVER UNION (IOU) OF TWO RECTANGLES
    # boxA and boxB are arrays of [x1,y1,x2,y2], where (x2,y2) is the lower right corner
    U = Union(boxA, boxB)
    if U <= 0:
        return 0
    else:
        return Intersection(boxA, boxB) / U
    
    
def Colors(n_colors):
    colors = []
    for i in range(n_colors):
        rgb = [random.randint(40,220),random.randint(100,220),random.randint(200,220)]
        random.shuffle(rgb)
        colors.append(tuple(rgb))
    return colors


def split_frame(frame):
    sf_stack = []
    for i in range(len(anchor_df)):
        s = int(anchor_df['size'][i])
        sf = frame[anchor_df['y1'][i]:anchor_df['y2'][i], anchor_df['x1'][i]:anchor_df['x2'][i], :]
        sf_rows, sf_cols, sf_channels = np.shape(sf)
        if sf_rows > s or sf_cols > s:
            sf = resize(sf, (s,s))
        sf_stack.append(sf)
    return sf_stack


def calc_anchors(overlap = None, outfile = None):
    print('\n[calcAnchors]: Calculating Search Boundaries...')
    anchor_df = []
    if overlap == None:
        overlap = 0
    nx = int(np.floor((FRAME_WIDTH-INPUT_WIDTH*overlap)/(INPUT_HEIGHT*(1-overlap))))
    ny = int(np.floor((FRAME_HEIGHT-INPUT_HEIGHT*overlap)/(INPUT_WIDTH*(1-overlap)))) - 1
    sf_no = 0
    for scale in SEARCH_SCALES:
        x1 = 0; y1 = 0
        scaled_height = round(scale * INPUT_HEIGHT)
        scaled_width = round(scale * INPUT_WIDTH)
        nx = int(np.floor((FRAME_WIDTH-scaled_width*overlap)/(scaled_width*(1-overlap))))
        ny = int(np.floor((FRAME_HEIGHT-scaled_height*overlap)/(scaled_height*(1-overlap))))
        for i in range(ny):
            y1 = y1 + round(scaled_height * (1-overlap))
            y2 = y1 + scaled_height
            for j in range(nx):
                x1 = j * round(scaled_width * (1-overlap))
                x2 = x1 + scaled_width
                if x2 > FRAME_WIDTH or y2 > FRAME_HEIGHT:
                    y2 = FRAME_HEIGHT
                    x2 = FRAME_WIDTH
                    x1 = x2 - scaled_width
                    y1 = y2 - scaled_height
                new_row = [sf_no, scaled_width, scaled_height, scale, x1, x2, y1, y2]
                anchor_df.append(new_row) 
                sf_no += 1
    anchor_df = pd.DataFrame(anchor_df, columns = ['subframe','scaled_width','scaled_height','scale','x1','x2','y1','y2'])
    if outfile != None:
        anchor_df.to_csv(outfile, index = False)
        print('\n[calcBoundaries]: Boundaries written to file.')
    return anchor_df


def fill_map(lyr, rad = 3):
    mod_lyr = np.zeros(lyr.shape)
    for i in range(lyr.shape[0]):
        for j in range(lyr.shape[1]):
            i_st = max(0, i-rad)
            i_end = min(lyr.shape[0], i+rad)
            j_st = max(0, j-rad)
            j_end = min(lyr.shape[1], j+rad)
            mod_lyr[i,j] = round(np.mean(lyr[i_st:i_end, j_st:j_end]))
    return mod_lyr


def draw_class_details(output, params, class_text = True, bar_chart = False):
    i = 0
    for val in pc:
        class_str = '{}% : {}'.format(int(100*val), classes[i])
        if class_text: cv2.putText(output, class_str, (C + spread, R - spread + 20*i), params['font'], 0.50, class_colors[i], 1, params['line_style'])
        if bar_chart: cv2.rectangle(output, (C + spread + 30, R - spread + 20*i - 20), (C + spread + 30 + int(100*val), R - spread + 20*i), class_colors[i], thickness = -1)
        i += 1
        
        
def hud(output):
    hud_labs = ['SubframeSize: ({},{})'.format(INPUT_HEIGHT, INPUT_WIDTH),
                'DetThresh:    {}'.format(DETECT_THRESH),
                'Alpha:        {}'.format(HUD_ALPHA),
                'FPS:          {} Hz'.format(FPS)]
    J = 0
    for lab in hud_labs:
        cv2.putText(output, lab, (5, 15 + J*15), HUD_FONT, 0.40, HUD_COLOR, 1, HUD_LINE_STYLE)
        J += 1
