import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2
import colorsys
import random
from PIL import Image, ImageDraw, ImageFont

def generate_colors(colors_count):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / colors_count, 1., 1.)
                  for x in range(colors_count)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

# Generate 256 colors for drawing bounding boxes.
COLORS_NUM = 256
rgb_colors = generate_colors(256)

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def normalize(image):
    image = image / 255.
    
    return image

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def draw_boxes(image, boxes, labels):
    
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image, 
                    #labels[box.get_label()] + ' ' + str(box.get_score()), 
                    labels[box.get_label()] + ' ' + "{:.2f}".format(box.get_score()), 
                    (xmin, ymin - 13), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 
                    cv2.FONT_HERSHEY_PLAIN,
                    1e-3 * image.shape[0], 
                    #(0,255,0), 2)
                    rgb_colors[box.get_label()%COLORS_NUM], 2)
        
    return image        

def draw_pil_image_boxes(image, boxes, labels):
    # set font
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                         size=np.floor(3e-2 * image.size[1]+0.5).astype('int32'))

    # compute appropriate thickness of line
    thickness = (image.size[0]+image.size[1]) // 300    

    for box in boxes:
        predicted_class = labels[box.get_label()]
        score = box.get_score()
        img_label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(img_label, font)

        xmin  = int((box.x - box.w/2) * image.size[0])
        xmax  = int((box.x + box.w/2) * image.size[0])
        ymin  = int((box.y - box.h/2) * image.size[1])
        ymax  = int((box.y + box.h/2) * image.size[1])

        top = int((box.y - box.h/2) *  image.size[1])
        left = int((box.x - box.w/2) * image.size[0])
        bottom = int((box.y + box.h/2) * image.size[1])
        right = int((box.x + box.w/2) * image.size[0])
        
        #top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # draw a rectangle for image bounding box
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=rgb_colors[box.get_label()%COLORS_NUM])

        # draw a rectangle as background for image label
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=rgb_colors[box.get_label()%COLORS_NUM])

        # draw image label text on image
        draw.text(text_origin, img_label, fill=(0, 0, 0), font=font)

        del draw
        
    return image

def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if classes.any():
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x, y, w, h, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)



