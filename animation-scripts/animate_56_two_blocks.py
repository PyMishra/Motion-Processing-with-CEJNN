# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:35:49 2020

@author: prate
"""

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
    img = np.zeros((height, width, channels), dtype = "uint8")
    img[:] = color
    return img

def create_object(height, width, channels, start1, end1, start2, end2):
    img = create_background(height, width, channels)
    color = (255, 0 , 0) # blue in cv
    img = cv2.rectangle(img, start1, end1, color, -1)
    img = cv2.rectangle(img, start2, end2, color, -1)
    return img

def create_video_right_two(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[0], end_loc[0], step):
        start1 = (centre - int(0.5*l), start_loc[1] - (height//4) - int(0.5*b))
        end1 = (centre + int(0.5*l), start_loc[1] - (height//4) + int(0.5*b))
        start2 = (centre - int(0.5*l), start_loc[1] + (height//4) - int(0.5*b))
        end2 = (centre + int(0.5*l), start_loc[1] + (height//4) + int(0.5*b))
        frame = create_object(height, width, channels, start1, end1, start2, end2)
        video.write(frame)
    
    video.release()

def create_video_left_up(video_name, height, width, channels, start_loc, end_loc, step=-2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[0], end_loc[0], step):
        start1 = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end1 = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        start2 = (start_loc[1] - int(0.5*l), centre - int(0.5*b))
        end2 = (start_loc[1] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start1, end1, start2, end2)
        video.write(frame)
    
    video.release()
    
def create_video_right_down(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[0], end_loc[0], step):
        start1 = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end1 = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        start2 = (start_loc[1] - int(0.5*l), centre - int(0.5*b))
        end2 = (start_loc[1] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start1, end1, start2, end2)
        video.write(frame)
    
    video.release()
    
def create_video_down_two(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[1], end_loc[1], step):
        start1 = (start_loc[0] - (height//4) - int(0.5*l), centre - int(0.5*b))
        end1 = (start_loc[0] - (height//4) + int(0.5*l), centre + int(0.5*b))
        start2 = (start_loc[0] + (height//4) - int(0.5*l), centre - int(0.5*b))
        end2 = (start_loc[0] + (height//4) + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start1, end1, start2, end2)
        video.write(frame)
    
    video.release()



height = 56; width = 56; channels = 3; l = 8; b = 8

c = 0
for i in range(1):
    
#    step = 2 + randint(-1, 1)
    
    video_name = './right_two_{}.mp4'.format(c)
    x1 = 14
    x2 = 42
    y = 28
    create_video_right_two(video_name, height, width, channels, (x1, y), (x2, y), 1, l, b)
    
    video_name = './left_up_{}.mp4'.format(c)
    x1 = 42
    x2 = 14
    y = 28
    create_video_left_up(video_name, height, width, channels, (x1, y), (x2, y), -1, l, b)
    
    video_name = './right_down_{}.mp4'.format(c)
    x1 = 14
    x2 = 42
    y = 28
    create_video_right_down(video_name, height, width, channels, (x1, y), (x2, y), 1, l, b)
    
    video_name = './down_two_{}.mp4'.format(c)
    x = 28
    y1 = 14
    y2 = 42
    create_video_down_two(video_name, height, width, channels, (x, y1), (x, y2), 1, l, b)
    
    c += 1
    

#video_name = './left.mp4'
#create_video_left(video_name, height, width, channels, (168, 112), (56, 112))

#video_name = './right.mp4'
#create_video_right_two(video_name, height, width, channels, (14, 28), (42, 28), 1, l, b)










