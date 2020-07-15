# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:18:10 2019

@author: Prateek
"""

import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from random import uniform, randint

def create_background(height, width, channels=3, color=(0, 0, 255)):
    img = np.zeros((height, width), dtype = "uint8")
#    img[:] = color
    return img

def create_object(height, width, channels, start, end):
    img = create_background(height, width, channels)
    color = (255, 0, 0) # blue in cv
    img = cv2.rectangle(img, start, end, color, -1) 
    return img

def create_video_right(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=10):
    fourcc = VideoWriter_fourcc(*'XVID')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height), 0)
    for centre in range(start_loc[0], end_loc[0], step):
        start = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()

def create_video_left(video_name, height, width, channels, start_loc, end_loc, step=-2, l=40, b=40, FPS=10):
    fourcc = VideoWriter_fourcc(*'XVID')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height), 0)
    for centre in range(start_loc[0], end_loc[0], step):
        start = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()
    
def create_video_up(video_name, height, width, channels, start_loc, end_loc, step=-2, l=40, b=40, FPS=10):
    fourcc = VideoWriter_fourcc(*'XVID')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height), 0)
    for centre in range(start_loc[1], end_loc[1], step):
        start = (start_loc[0] - int(0.5*l), centre - int(0.5*b))
        end = (start_loc[0] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()
    
def create_video_down(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=10):
    fourcc = VideoWriter_fourcc(*'XVID')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height), 0)
    for centre in range(start_loc[1], end_loc[1], step):
        start = (start_loc[0] - int(0.5*l), centre - int(0.5*b))
        end = (start_loc[0] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()



height = 28; width = 28; channels = 3; l = 4; b = 4

c = 0
for i in range(5):
    
#    step = 2 + randint(-1, 1)
    
    video_name = './right_{}.avi'.format(c)
    x1 = 7
    x2 = int(21 + uniform(-1, 1) * l)
    y = 14
    create_video_right(video_name, height, width, channels, (x1, y), (x2, y), 1, l, b)
    
    video_name = './left_{}.mp4'.format(c)
    x1 = 21
    x2 = int(7 + uniform(-1, 1) * l)
    y = 14
    create_video_left(video_name, height, width, channels, (x1, y), (x2, y), -1, l, b)
    
    video_name = './up_{}.mp4'.format(c)
    x = 14
    y1 = 21
    y2 = int(7 + uniform(-1, 1) * b)
    create_video_up(video_name, height, width, channels, (x, y1), (x, y2), -1, l, b)
    
    video_name = './down_{}.mp4'.format(c)
    x = 14
    y1 = 7
    y2 = int(21 + uniform(-1, 1) * b)
    create_video_down(video_name, height, width, channels, (x, y1), (x, y2), 1, l, b)
    
    c += 1
    

#video_name = './left.mp4'
#create_video_left(video_name, height, width, channels, (168, 112), (56, 112))













