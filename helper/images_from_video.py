# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:55:03 2019

@author: Prateek
"""
import os
import handle_video as hv

action_folders = os.listdir(r'D:\Prateek\Data\UCF11_updated_mpg')

for folder in action_folders:
    action_list = os.listdir(r'D:\Prateek\Data\UCF11_updated_mpg\{}'.format(folder))[1:]
    
    for action in action_list:
        videos = os.listdir(r'D:\Prateek\Data\UCF11_updated_mpg\{}\{}'.format(folder, action))
        
        for video in videos:
            video_path = r'D:\Prateek\Data\UCF11_updated_mpg\{}\{}\{}'.format(folder, action, video)
            image_folder = r'D:\Prateek\Data\UCF11\{}\{}'.format(folder, video[:-4])
            print(video_path)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            hv.get_image_folder(image_folder, video_path)