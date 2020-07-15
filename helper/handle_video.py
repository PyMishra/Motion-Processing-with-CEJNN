# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:07:40 2019

@author: Prateek
"""

import cv2
import math

def capture_frames(video, frame_rate_needed, image_folder, label):
    while video.isOpened():
        frameid = video.get(1)
        ret, frame = video.read()
        if ret != True:
            break
        if frame_rate_needed == 0:
            filename = image_folder + '\{}_'.format(label) + str(int(frameid)) + '.jpg'
            cv2.imwrite(filename, frame)
        else:
            if frameid % math.floor(frame_rate_needed) == 0:
                filename = image_folder + '\{}_'.format(label) + str(int(frameid)) + '.jpg'
                cv2.imwrite(filename, frame)
    
def count_frames(video):
    # To find the number of frames and length of video
    total_frames = (video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = (video.get(cv2.CAP_PROP_FPS))
    return total_frames, frame_rate


def get_frames_from_video(video, frames_needed):
    if frames_needed == 0:
        return 0        
    total_frames, frame_rate = count_frames(video)
    frame_rate_needed = total_frames / frames_needed
    return frame_rate_needed

def get_image_folder(image_folder, video_path, label):
    # Create a video capture object and call functions to generate images
    video = cv2.VideoCapture(video_path)
    frames_needed = 0
    frame_rate_needed = get_frames_from_video(video, frames_needed)
    if int(frame_rate_needed) or frame_rate_needed == 0:
        capture_frames(video, frame_rate_needed, image_folder, label)
    video.release()
    
#image_folder = r'D:\Prateek\Data\images'
#video_path = r'D:\Prateek\Data\UCF11_combined\basketball_1.mpg'
#
#get_image_folder(image_folder, video_path, 'shooting')