# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:18:10 2019

@author: Prateek
"""

import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from random import uniform

def create_background(height, width, channels=3, color=(0, 0, 255)):
    img = np.zeros((height, width, channels), dtype = "uint8")
    img[:] = color
    return img

def create_object(height, width, channels, start, end):
    img = create_background(height, width, channels)
    color = (255, 0 , 0) # blue in cv
    img = cv2.rectangle(img, start, end, color, -1) 
    return img

def create_video_right(video_name, height, width, channels, start_loc, end_loc, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[0], end_loc[0], 2):
        start = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()

def create_video_left(video_name, height, width, channels, start_loc, end_loc, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[0], end_loc[0], -2):
        start = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()
    
def create_video_up(video_name, height, width, channels, start_loc, end_loc, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[1], end_loc[1], -2):
        start = (start_loc[0] - int(0.5*l), centre - int(0.5*b))
        end = (start_loc[0] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()
    
def create_video_down(video_name, height, width, channels, start_loc, end_loc, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[1], end_loc[1], 2):
        start = (start_loc[0] - int(0.5*l), centre - int(0.5*b))
        end = (start_loc[0] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()



height = 224; width = 224; channels = 3; l = 30; b = 30

c = 0
for i in range(10):
    
    video_name = './right_{}.mp4'.format(c)
    x1 = 56
    x2 = int(168 + uniform(-1, 1) * l)
    y = 112
    create_video_right(video_name, height, width, channels, (x1, y), (x2, y))
    
    video_name = './left_{}.mp4'.format(c)
    x1 = 168
    x2 = int(56 + uniform(-1, 1) * l)
    y = 112
    create_video_left(video_name, height, width, channels, (x1, y), (x2, y))
    
    video_name = './up_{}.mp4'.format(c)
    x = 112
    y1 = 168
    y2 = int(56 + uniform(-1, 1) * b)
    create_video_up(video_name, height, width, channels, (x, y1), (x, y2))
    
    video_name = './down_{}.mp4'.format(c)
    x = 112
    y1 = 56
    y2 = int(168 + uniform(-1, 1) * b)
    create_video_down(video_name, height, width, channels, (x, y1), (x, y2))
    
    c += 1
    

#video_name = './left.mp4'
#create_video_left(video_name, height, width, channels, (168, 112), (56, 112))













