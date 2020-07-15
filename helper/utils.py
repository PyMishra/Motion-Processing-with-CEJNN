# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:27:56 2019

@author: Prateek
"""

import tensorflow as tf
from keras.utils import to_categorical

# Function to save session weights
def save_session(path, sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path)
    print('Session saved in file: {}'.format(save_path))
    
# Function to restore weights
def restore_session(path, sess):
    saver = tf.train.Saver()
    saver.restore(sess, path)
    print('Session Restored')
    
# Create labels for videos, will be the first word. Return one hot labels and a dictionary
def create_labels_for_videos(video_list, classes=11):
    # Get video count
    c = 0
    video_dict = {}
    no_of_videos = len(video_list)
    labels = []
    for i in range(no_of_videos):
        name = video_list[i]
        first_name = name.split('_')[0]
        
        if first_name not in video_dict:
            video_dict[first_name] = c
            c += 1
        
        labels.append(video_dict[first_name])
    labels = to_categorical(labels, classes)
    return video_dict, labels
