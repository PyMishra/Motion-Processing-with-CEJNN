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

def create_object(height, width, channels, centre, radius):
    img = create_background(height, width, channels)
    color = (255, 0 , 0) # blue in cv
    img = cv2.circle(img, centre, radius, color, -1) 
    return img

def create_video_right(video_name, height, width, channels, start_loc, end_loc, step=2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for x in range(start_loc[0], end_loc[0], step):
        centre = (x, start_loc[1])
        frame = create_object(height, width, channels, centre, radius)
        video.write(frame)
    
    video.release()

def create_video_left(video_name, height, width, channels, start_loc, end_loc, step=-2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for x in range(start_loc[0], end_loc[0], step):
        centre = (x, start_loc[1])
        frame = create_object(height, width, channels, centre, radius)
        video.write(frame)
    
    video.release()
    
def create_video_up(video_name, height, width, channels, start_loc, end_loc, step=-2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for y in range(start_loc[1], end_loc[1], step):
        centre = (start_loc[0], y)
        frame = create_object(height, width, channels, centre, radius)
        video.write(frame)
    
    video.release()
    
def create_video_down(video_name, height, width, channels, start_loc, end_loc, step=2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    for y in range(start_loc[1], end_loc[1], step):
        centre = (start_loc[0], y)
        frame = create_object(height, width, channels, centre, radius)
        video.write(frame)
    
    video.release()
    
def create_bot_right_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] < end_loc[0]:
        frame = create_object(height, width, channels, tuple(centre), radius)
        video.write(frame)
        centre[0] += step
        centre[1] += step
    
    video.release()
    
def create_bot_left_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] > end_loc[0]:
        frame = create_object(height, width, channels, tuple(centre), radius)
        video.write(frame)
        centre[0] -= step
        centre[1] += step
    
    video.release()
    
def create_top_left_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] > end_loc[0]:
        frame = create_object(height, width, channels, tuple(centre), radius)
        video.write(frame)
        centre[0] -= step
        centre[1] -= step
    
    video.release()

def create_top_right_diagnol(video_name, height, width, channels, start_loc, end_loc, step=2, radius=4, FPS=20):
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(video_name, fourcc, float(FPS), (width, height))
    centre = [start_loc[0], start_loc[1]]
    while centre[0] < end_loc[0]:
        frame = create_object(height, width, channels, tuple(centre), radius)
        video.write(frame)
        centre[0] += step
        centre[1] -= step
    
    video.release()

    



height = 56; width = 56; channels = 3; radius = 4

c = 0
for i in range(2):
    
#    step = 2 + randint(-1, 1)
    
    video_name = './right_{}.mp4'.format(c)
    x1 = 14
    x2 = int(42 + uniform(-1, 1) * radius)
    y = 28
    create_video_right(video_name, height, width, channels, (x1, y), (x2, y), 1, radius)
    
    video_name = './left_{}.mp4'.format(c)
    x1 = 42
    x2 = int(14 + uniform(-1, 1) * radius)
    y = 28
    create_video_left(video_name, height, width, channels, (x1, y), (x2, y), -1, radius)
    
    video_name = './up_{}.mp4'.format(c)
    x = 28
    y1 = 42
    y2 = int(14 + uniform(-1, 1) * radius)
    create_video_up(video_name, height, width, channels, (x, y1), (x, y2), -1, radius)
    
    video_name = './down_{}.mp4'.format(c)
    x = 28
    y1 = 14
    y2 = int(42 + uniform(-1, 1) * radius)
    create_video_down(video_name, height, width, channels, (x, y1), (x, y2), 1, radius)
    
    video_name = './bot_right_{}.mp4'.format(c)
    x1, y1 = 14, 14
    variation = uniform(-1, 1) * radius
    x2 = int(42 + variation)
    y2 = int(42 + variation)
    create_bot_right_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, radius)
    
    video_name = './bot_left_{}.mp4'.format(c)
    x1, y1 = 42, 14
    variation = uniform(-1, 1) * radius
    x2 = int(14 + variation)
    y2 = int(42 + variation)
    create_bot_left_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, radius)
    
    video_name = './top_left_{}.mp4'.format(c)
    x1, y1 = 42, 42
    variation = uniform(-1, 1) * radius
    x2 = int(14 + variation)
    y2 = int(14 + variation)
    create_top_left_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, radius)
    
    video_name = './top_right_{}.mp4'.format(c)
    x1, y1 = 14, 42
    variation = uniform(-1, 1) * radius
    x2 = int(42 + variation)
    y2 = int(14 + variation)
    create_top_right_diagnol(video_name, height, width, channels, (x1, y1), (x2, y2), 1, radius)
    
    c += 1
    

#video_name = './bot_right_diag.mp4'
#create_bot_right_diagnol(video_name, height, width, channels, (14, 14), (42, 42), 1, l, b)
