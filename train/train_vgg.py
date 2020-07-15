# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:41:15 2019

@author: Prateek
"""
import os
import datetime
import shutil as sh
import tensorflow as tf
import numpy as np
import model_vgg_all_jordan
import handle_video_vgg as hv
import utils
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from progressbar import ProgressBar
import random

# List directories for video and temporary image folder
video_folder = r'D:\Prateek\Data\UCF11_combined'
temp_folder = r'D:\Prateek\Data\temp'

#os.mkdir(temp_folder)

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d_(%H-%M-%S)")

# Get the list and count of videos in the directory
video_list = os.listdir(video_folder)
random.shuffle(video_list)
video_count = len(video_list)
video_list = video_list[:int(0.75*video_count)] # Comment for all videos

# Create a train, valid and test split of the data
video_dict, y_labels = utils.create_labels_for_videos(video_list)
train_video_list, valid_video_list, y_train_labels, y_valid_labels = train_test_split(video_list, y_labels,
                                                                                      test_size=0.25, stratify=y_labels, random_state=42)
valid_video_list, test_video_list, y_valid_labels, y_test_labels = train_test_split(valid_video_list, y_valid_labels,
                                                                                    test_size=0.4, stratify=y_valid_labels, random_state=42)

# Set basic parameters
batch_size = 1 # Very important to keep it 1
lr = 0.000001
epochs = 10

# Inititalize network
network = model_vgg_all_jordan.Network()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# Try to restore weights, if they exist
exp_path =  r'D:\Prateek\Experiments\{}-(VGG)'.format(current_time)

try:
    weights_path = os.path.join(os.listdir(os.path.dirname(exp_path))[-1], 'weights')
    restore_path = os.path.dirname(exp_path)  + '/' + os.listdir(os.path.dirname(exp_path))[-1]
    weights_path = restore_path + '/' + 'weights/'
except IndexError:
    weights_path = []
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    print('Weights Directory not present, training from scratch!')

if weights_path == []:
    pass
else:
#    utils.restore_session(weights_path, sess) # Comment for fresh session
    pass

# Create results directory
res_path = exp_path + '/' + 'results/'
if not os.path.exists(res_path):
    os.makedirs(res_path)
    print('Making Results (Res Path) Directory!')

# Create weights directory
weights_path = exp_path + '/' + 'weights/'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print('Making Weights Directory!')
    
# Create image path
output_image_path = exp_path +  '/' + 'images/'
if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)
    print('Making Image Directory!')
    

# Open log file
filename = '/SessLog_Epochs{}_Time_{}.txt'.format(epochs, current_time)
log = open(res_path + filename, 'w')
log.write('Experiment 6: CNN all elman and one jordan. Frames = 50')
log.write('\nLearning Rate:{}\n'.format(lr))
log.close()

for epoch in range(epochs):

    log = open(res_path + filename, 'a')    
    print('\nEpoch_{}\n'.format(epoch))
    train_loss_per_epoch = 0
    pbar_track = ProgressBar()
    
    y_pred = []
    ################################################################### Training the network ############################################################
    for i in pbar_track(range(len(train_video_list))):
        
        # Delete and create fresh directory
        sh.rmtree(temp_folder, ignore_errors=True)
        os.mkdir(temp_folder)
        
        # Join video path with folder to get a video to generate frames from
        video = train_video_list[i]
        video_path = os.path.join(video_folder, video)
#        print(video, video.split('_')[0])
        hv.get_image_folder(temp_folder, video_path, video.split('_')[0])
        
        # Get the label
        curr_label = y_train_labels[i].reshape(-1, 11)
        
        # Get the list of images in the folder
        image_list = os.listdir(temp_folder)
        no_of_frames = len(image_list)
        
        # Initialize loop variables
        eloop_11 = np.ones([batch_size, 224, 224, 64])
        eloop_12 = np.ones([batch_size, 224, 224, 64])
        eloop_21 = np.ones([batch_size, 112, 112, 128])
        eloop_22 = np.ones([batch_size, 112, 112, 128])
        eloop_31 = np.ones([batch_size, 56, 56, 256])
        eloop_32 = np.ones([batch_size, 56, 56, 256])
        eloop_41 = np.ones([batch_size, 28, 28, 512])
        eloop_42 = np.ones([batch_size, 28, 28, 512])
        eloop_51 = np.ones([batch_size, 14, 14, 512])
        eloop_52 = np.ones([batch_size, 14, 14, 512])
        
        jloop_11 = np.ones([batch_size, 224, 224, 64])
        
        loss_per_video = 0
        
        for image in image_list:
            # Get one image frame and pass it to network
            image_path = os.path.join(temp_folder, image)
            img = cv2.imread(image_path)
            
            eloop_11_, eloop_12_, eloop_21_, eloop_22_, eloop_31_, eloop_32_, eloop_41_, eloop_42_, eloop_51_, eloop_52_, jloop_11_, cost, optimizer, output = sess.run([                                                        network['conv11_in'],
                                                        network['conv12_in'],
                                                        network['conv21_in'],
                                                        network['conv22_in'],
                                                        network['conv31_in'],
                                                        network['conv32_in'],
                                                        network['conv41_in'],
                                                        network['conv42_in'],
                                                        network['conv51_in'],
                                                        network['conv52_in'],
                                                        network['deconv5_in'],
                                                        network['cost'],
                                                        network['optimizer'],
                                                        network['output']],
                                                feed_dict={network['image_input']: np.reshape(img, [batch_size, 224, 224, 3]),
                                                           network['conv11_eloop_input']: eloop_11,
                                                           network['conv12_eloop_input']: eloop_12,
                                                           network['conv21_eloop_input']: eloop_21,
                                                           network['conv22_eloop_input']: eloop_22,
                                                           network['conv31_eloop_input']: eloop_31,
                                                           network['conv32_eloop_input']: eloop_32,
                                                           network['conv41_eloop_input']: eloop_41,
                                                           network['conv42_eloop_input']: eloop_42,
                                                           network['conv51_eloop_input']: eloop_51,
                                                           network['conv52_eloop_input']: eloop_52,
                                                           network['conv11_jloop_input']: jloop_11,
                                                           network['lr']: lr,
                                                           network['label']: curr_label})
    
            # Update context layers
            eloop_11 = eloop_11_
            eloop_12 = eloop_12_
            eloop_21 = eloop_21_
            eloop_22 = eloop_22_
            eloop_31 = eloop_31_
            eloop_32 = eloop_32_
            eloop_41 = eloop_41_
            eloop_42 = eloop_42_
            eloop_51 = eloop_51_
            eloop_52 = eloop_52_
            jloop_11 = jloop_11_
            
            loss_per_video += cost
            
        train_loss_per_epoch += loss_per_video
        y_pred.append(output[0])
    
    # Calculate accuracy        
    y_pred = np.array(y_pred)
    train_corr_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_train_labels, 1))
    acc = tf.reduce_mean(tf.cast(train_corr_pred, tf.float32))
    train_accuracy = sess.run(acc)
    
    print('Epoch training loss:', train_loss_per_epoch)
    # Save weights
    utils.save_session(weights_path, sess)
    print('Current Epoch is {}, weights saved'.format(epoch))
    
    ####################################################### Validating the network #################################################################
    valid_loss_per_epoch = 0
    pbar_track = ProgressBar()
    y_pred = []
    for i in pbar_track(range(len(valid_video_list))):
        
        # Delete and create fresh directory
        sh.rmtree(temp_folder, ignore_errors=True)
        os.mkdir(temp_folder)
        
        # Join video path with folder to get a video to generate frames from
        video = valid_video_list[i]
        video_path = os.path.join(video_folder, video)
#        print(video, video.split('_')[0])
        hv.get_image_folder(temp_folder, video_path, video.split('_')[0])
        
        # Get the label
        curr_label = y_valid_labels[i].reshape(-1, 11)
        
        # Get the list of images in the folder
        image_list = os.listdir(temp_folder)
        no_of_frames = len(image_list)
        
        # Initialize loop variables
        eloop_11 = np.ones([batch_size, 224, 224, 64])
        eloop_12 = np.ones([batch_size, 224, 224, 64])
        eloop_21 = np.ones([batch_size, 112, 112, 128])
        eloop_22 = np.ones([batch_size, 112, 112, 128])
        eloop_31 = np.ones([batch_size, 56, 56, 256])
        eloop_32 = np.ones([batch_size, 56, 56, 256])
        eloop_41 = np.ones([batch_size, 28, 28, 512])
        eloop_42 = np.ones([batch_size, 28, 28, 512])
        eloop_51 = np.ones([batch_size, 14, 14, 512])
        eloop_52 = np.ones([batch_size, 14, 14, 512])
        
        jloop_11 = np.ones([batch_size, 224, 224, 64])
        
        loss_per_video = 0
        
        for image in image_list:
            # Get one image frame and pass it to network
            image_path = os.path.join(temp_folder, image)
            img = cv2.imread(image_path)
            
            eloop_11_, eloop_12_, eloop_21_, eloop_22_, eloop_31_, eloop_32_, eloop_41_, eloop_42_, eloop_51_, eloop_52_, jloop_11_, cost, output = sess.run([                                                        network['conv11_in'],
                                                        network['conv12_in'],
                                                        network['conv21_in'],
                                                        network['conv22_in'],
                                                        network['conv31_in'],
                                                        network['conv32_in'],
                                                        network['conv41_in'],
                                                        network['conv42_in'],
                                                        network['conv51_in'],
                                                        network['conv52_in'],
                                                        network['deconv5_in'],
                                                        network['cost'],
                                                        network['output']],
                                                feed_dict={network['image_input']: np.reshape(img, [batch_size, 224, 224, 3]),
                                                           network['conv11_eloop_input']: eloop_11,
                                                           network['conv12_eloop_input']: eloop_12,
                                                           network['conv21_eloop_input']: eloop_21,
                                                           network['conv22_eloop_input']: eloop_22,
                                                           network['conv31_eloop_input']: eloop_31,
                                                           network['conv32_eloop_input']: eloop_32,
                                                           network['conv41_eloop_input']: eloop_41,
                                                           network['conv42_eloop_input']: eloop_42,
                                                           network['conv51_eloop_input']: eloop_51,
                                                           network['conv52_eloop_input']: eloop_52,
                                                           network['conv11_jloop_input']: jloop_11,
                                                           network['lr']: lr,
                                                           network['label']: curr_label})
    
            # Update context layers
            eloop_11 = eloop_11_
            eloop_12 = eloop_12_
            eloop_21 = eloop_21_
            eloop_22 = eloop_22_
            eloop_31 = eloop_31_
            eloop_32 = eloop_32_
            eloop_41 = eloop_41_
            eloop_42 = eloop_42_
            eloop_51 = eloop_51_
            eloop_52 = eloop_52_
            jloop_11 = jloop_11_
            
            loss_per_video += cost
            
        valid_loss_per_epoch += loss_per_video
        y_pred.append(output[0])
    
    # Calculate accuracy        
    y_pred = np.array(y_pred)
    valid_corr_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_valid_labels, 1))
    acc = tf.reduce_mean(tf.cast(valid_corr_pred, tf.float32))
    valid_accuracy = sess.run(acc)
    cm = confusion_matrix(y_valid_labels.argmax(axis=1), y_pred.argmax(axis=1), list(range(11)))
    print('\n')
    print(cm)
    
    #################################################################### Testing the network #####################################################
    
    test_loss_per_epoch = 0
    pbar_track = ProgressBar()
    y_pred = []
#    output_image_folder = output_image_path + '/' + 'Epoch_{}'.format(epoch)
#    if not os.path.exists(output_image_folder):
#        os.makedirs(output_image_folder)
        
    for i in pbar_track(range(len(test_video_list))):
        
        # Delete and create fresh directory
        sh.rmtree(temp_folder, ignore_errors=True)
        os.mkdir(temp_folder)
        
        # Join video path with folder to get a video to generate frames from
        video = test_video_list[i]        
        video_path = os.path.join(video_folder, video)
#        print(video, video.split('_')[0])
        hv.get_image_folder(temp_folder, video_path, video.split('_')[0])
        
        # Make directory for this video in image path
#        image_folder = output_image_folder + '/' + video[:-4]
#        if not os.path.exists(image_folder):
#            os.makedirs(image_folder)
    
        
        # Get the label
        curr_label = y_test_labels[i].reshape(-1, 11)
        
        # Get the list of images in the folder
        image_list = os.listdir(temp_folder)
        no_of_frames = len(image_list)
        
        # Initialize loop variables
        eloop_11 = np.ones([batch_size, 224, 224, 64])
        eloop_12 = np.ones([batch_size, 224, 224, 64])
        eloop_21 = np.ones([batch_size, 112, 112, 128])
        eloop_22 = np.ones([batch_size, 112, 112, 128])
        eloop_31 = np.ones([batch_size, 56, 56, 256])
        eloop_32 = np.ones([batch_size, 56, 56, 256])
        eloop_41 = np.ones([batch_size, 28, 28, 512])
        eloop_42 = np.ones([batch_size, 28, 28, 512])
        eloop_51 = np.ones([batch_size, 14, 14, 512])
        eloop_52 = np.ones([batch_size, 14, 14, 512])
        
        jloop_11 = np.ones([batch_size, 224, 224, 64])
        
        loss_per_video = 0
        
        for image in image_list:
            # Get one image frame and pass it to network
            image_path = os.path.join(temp_folder, image)
            img = cv2.imread(image_path)
            
            eloop_11_, eloop_12_, eloop_21_, eloop_22_, eloop_31_, eloop_32_, eloop_41_, eloop_42_, eloop_51_, eloop_52_, jloop_11_, cost, output = sess.run([                                                        network['conv11_in'],
                                                        network['conv12_in'],
                                                        network['conv21_in'],
                                                        network['conv22_in'],
                                                        network['conv31_in'],
                                                        network['conv32_in'],
                                                        network['conv41_in'],
                                                        network['conv42_in'],
                                                        network['conv51_in'],
                                                        network['conv52_in'],
                                                        network['deconv5_in'],
                                                        network['cost'],
                                                        network['output']],
                                                feed_dict={network['image_input']: np.reshape(img, [batch_size, 224, 224, 3]),
                                                           network['conv11_eloop_input']: eloop_11,
                                                           network['conv12_eloop_input']: eloop_12,
                                                           network['conv21_eloop_input']: eloop_21,
                                                           network['conv22_eloop_input']: eloop_22,
                                                           network['conv31_eloop_input']: eloop_31,
                                                           network['conv32_eloop_input']: eloop_32,
                                                           network['conv41_eloop_input']: eloop_41,
                                                           network['conv42_eloop_input']: eloop_42,
                                                           network['conv51_eloop_input']: eloop_51,
                                                           network['conv52_eloop_input']: eloop_52,
                                                           network['conv11_jloop_input']: jloop_11,
                                                           network['lr']: lr,
                                                           network['label']: curr_label})
    
            # Update context layers
            eloop_11 = eloop_11_
            eloop_12 = eloop_12_
            eloop_21 = eloop_21_
            eloop_22 = eloop_22_
            eloop_31 = eloop_31_
            eloop_32 = eloop_32_
            eloop_41 = eloop_41_
            eloop_42 = eloop_42_
            eloop_51 = eloop_51_
            eloop_52 = eloop_52_
            jloop_11 = jloop_11_
            
            loss_per_video += cost
            
        test_loss_per_epoch += loss_per_video
        y_pred.append(output[0])
    
    # Calculate accuracy        
    y_pred = np.array(y_pred)
    test_corr_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test_labels, 1))
    acc = tf.reduce_mean(tf.cast(test_corr_pred, tf.float32))
    test_accuracy = sess.run(acc)
    
    # Write logs
    log.write('Epoch:{}, Training_Loss:{}, Validation_Loss:{}, Testing_Loss:{}, Training_Accuracy:{}, Validation_Accuracy:{}, Testing_Accuracy:{}'.format(
                    epoch, train_loss_per_epoch, valid_loss_per_epoch, test_loss_per_epoch, train_accuracy, valid_accuracy, test_accuracy))
    log.write('\n')
    print('Epoch:{}, Training_Loss:{}, Validation_Loss:{}, Testing_Loss:{}, Training_Accuracy:{}, Validation_Accuracy:{}, Testing_Accuracy:{}'.format(
                    epoch, train_loss_per_epoch, valid_loss_per_epoch, test_loss_per_epoch, train_accuracy, valid_accuracy, test_accuracy))
    log.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    