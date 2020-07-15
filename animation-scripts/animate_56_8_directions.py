# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:30:10 2020

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

def create_object(height, width, channels, start, end):
    img = create_background(height, width, channels)
    color = (255, 0 , 0) # blue in cv
    img = cv2.rectangle(img, start, end, color, -1) 
    return img

def create_video_right(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[0], end_loc[0], step):
        start = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()

def create_video_left(video_name, height, width, channels, start_loc, end_loc, step=-2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[0], end_loc[0], step):
        start = (centre - int(0.5*l), start_loc[1] - int(0.5*b))
        end = (centre + int(0.5*l), start_loc[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()
    
def create_video_up(video_name, height, width, channels, start_loc, end_loc, step=-2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[1], end_loc[1], step):
        start = (start_loc[0] - int(0.5*l), centre - int(0.5*b))
        end = (start_loc[0] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()
    
def create_video_down(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for centre in range(start_loc[1], end_loc[1], step):
        start = (start_loc[0] - int(0.5*l), centre - int(0.5*b))
        end = (start_loc[0] + int(0.5*l), centre + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
    
    video.release()
    
def create_bot_right_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] < end_loc[0]:
        start = (centre[0] - int(0.5*l), centre[1] - int(0.5*b))
        end = (centre[0] + int(0.5*l), centre[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
        centre[0] += step
        centre[1] += step
    
    video.release()
    
def create_bot_left_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] > end_loc[0]:
        start = (centre[0] - int(0.5*l), centre[1] - int(0.5*b))
        end = (centre[0] + int(0.5*l), centre[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
        centre[0] -= step
        centre[1] += step
    
    video.release()
    
def create_top_left_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] > end_loc[0]:
        start = (centre[0] - int(0.5*l), centre[1] - int(0.5*b))
        end = (centre[0] + int(0.5*l), centre[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
        centre[0] -= step
        centre[1] -= step
    
    video.release()

def create_top_right_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, l=40, b=40, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] < end_loc[0]:
        start = (centre[0] - int(0.5*l), centre[1] - int(0.5*b))
        end = (centre[0] + int(0.5*l), centre[1] + int(0.5*b))
        frame = create_object(height, width, channels, start, end)
        video.write(frame)
        centre[0] += step
        centre[1] -= step
    
    video.release()

    



height = 56; width = 56; channels = 3; l = 8; b = 8

c = 0
for i in range(1):
    
#    step = 2 + randint(-1, 1)
    
    video_name = './right_{}.mp4'.format(c)
    x1 = 14
    x2 = int(42 + uniform(-1, 1) * l)
    y = 28
    create_video_right(video_name, height, width, channels, (x1, y), (x2, y), 1, l, b)
    
    video_name = './left_{}.mp4'.format(c)
    x1 = 42
    x2 = int(14 + uniform(-1, 1) * l)
    y = 28
    create_video_left(video_name, height, width, channels, (x1, y), (x2, y), -1, l, b)
    
    video_name = './up_{}.mp4'.format(c)
    x = 28
    y1 = 42
    y2 = int(14 + uniform(-1, 1) * b)
    create_video_up(video_name, height, width, channels, (x, y1), (x, y2), -1, l, b)
    
    video_name = './down_{}.mp4'.format(c)
    x = 28
    y1 = 14
    y2 = int(42 + uniform(-1, 1) * b)
    create_video_down(video_name, height, width, channels, (x, y1), (x, y2), 1, l, b)
    
    video_name = './bot_right_{}.mp4'.format(c)
    x1, y1 = 14, 14
    variation = uniform(-1, 1) * l
    x2 = int(42 + variation)
    y2 = int(42 + variation)
    create_bot_right_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, l, b)
    
    video_name = './bot_left_{}.mp4'.format(c)
    x1, y1 = 42, 14
    variation = uniform(-1, 1) * l
    x2 = int(14 + variation)
    y2 = int(42 + variation)
    create_bot_left_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, l, b)
    
    video_name = './top_left_{}.mp4'.format(c)
    x1, y1 = 42, 42
    variation = uniform(-1, 1) * l
    x2 = int(14 + variation)
    y2 = int(14 + variation)
    create_top_left_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, l, b)
    
    video_name = './top_right_{}.mp4'.format(c)
    x1, y1 = 14, 42
    variation = uniform(-1, 1) * l
    x2 = int(42 + variation)
    y2 = int(14 + variation)
    create_top_right_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, l, b)
    
    c += 1
    

#video_name = './bot_right_diag.mp4'
#create_bot_right_diagnol(video_name, height, width, channels, (14, 14), (42, 42), 1, l, b)
