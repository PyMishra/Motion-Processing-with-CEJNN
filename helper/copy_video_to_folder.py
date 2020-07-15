# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:50:17 2019

@author: Prateek
"""

import os
from shutil import copy
import cv2

combined_folder = r'D:\Prateek\Data\UCF11_combined'
action_folders = os.listdir(r'D:\Prateek\Data\UCF11_updated_mpg')

if not os.path.exists(combined_folder):
    os.makedirs(combined_folder)

for folder in action_folders:
    action_list = os.listdir(r'D:\Prateek\Data\UCF11_updated_mpg\{}'.format(folder))[1:]
    
    c = 1
    for action in action_list:
        videos = os.listdir(r'D:\Prateek\Data\UCF11_updated_mpg\{}\{}'.format(folder, action))
           
        for video in videos:
            video_path = r'D:\Prateek\Data\UCF11_updated_mpg\{}\{}\{}'.format(folder, action, video)
            vcap = cv2.VideoCapture(video_path)
            if vcap.isOpened():
                width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # convert to int
                height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # convert to int
            
            if width == 320 and height == 240:            
                copy(video_path, combined_folder + '\{}_{}.mpg'.format(folder, str(c)))
                c += 1

print('Files copied successfully')