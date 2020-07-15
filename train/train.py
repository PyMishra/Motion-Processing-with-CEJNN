# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:21:17 2019

@author: Prateek
"""

import os
import datetime
import shutil as sh
import tensorflow as tf
import numpy as np
import model
import handle_video as hv
import utils
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from progressbar import ProgressBar

# List directories for video and temporary image folder
video_folder = r'D:\Prateek\Data\UCF11_combined'
temp_folder = r'D:\Prateek\Data\temp_images'

#os.mkdir(temp_folder)

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d_(%H-%M-%S)")

# Get the list and count of videos in the directory
video_list = os.listdir(video_folder)
video_count = len(video_list)

# Create a train, valid and test split of the data
video_dict, y_labels = utils.create_labels_for_videos(video_list)
train_video_list, valid_video_list, y_train_labels, y_valid_labels = train_test_split(video_list, y_labels,
                                                                                      test_size=0.25, stratify=y_labels, random_state=42)
valid_video_list, test_video_list, y_valid_labels, y_test_labels = train_test_split(valid_video_list, y_valid_labels,
                                                                                    test_size=0.4, stratify=y_valid_labels, random_state=42)

# Set basic parameters
batch_size = 1 # Very important to keep it 1
lr = 0.00000012
epochs = 10

# Inititalize network
network = model.Network()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Try to restore weights, if they exist
exp_path =  r'D:\Prateek\Experiments\{}'.format(current_time)

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
log.write('Experiment 6: CNN all elman and one jordan. Frames = max')
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
        eloop_1 = np.ones([batch_size, 60, 80, 96])
        eloop_2 = np.ones([batch_size, 15, 20, 256])
        eloop_3 = np.ones([batch_size, 15, 20, 384])
        eloop_4 = np.ones([batch_size, 15, 20, 384])
        eloop_5 = np.ones([batch_size, 15, 20, 256])
        jloop_1 = np.ones([batch_size, 60, 80, 96])
        
        loss_per_video = 0
        
        for image in image_list:
            # Get one image frame and pass it to network
            image_path = os.path.join(temp_folder, image)
            img = cv2.imread(image_path)
            
            eloop_1_, eloop_2_, eloop_3_, eloop_4_, eloop_5_, jloop_1_, cost, optimizer, output = sess.run([
                                            network['conv1_in'],
                                            network['conv2_in'],
                                            network['conv3_in'],
                                            network['conv4_in'],
                                            network['conv5_in'],
                                            network['deconv5_in'],
                                            network['cost'],
                                            network['optimizer'],
                                            network['output']],
                                        feed_dict={network['image_input']: np.reshape(img, [batch_size, 240, 320, 3]),
                                                   network['conv1_eloop_input']: eloop_1,
                                                   network['conv2_eloop_input']: eloop_2,
                                                   network['conv3_eloop_input']: eloop_3,
                                                   network['conv4_eloop_input']: eloop_4,
                                                   network['conv5_eloop_input']: eloop_5,
                                                   network['conv1_jloop_input']: jloop_1,
                                                   network['lr']: lr,
                                                   network['label']: curr_label})
            
            # Update context layers
            eloop_1 = eloop_1_
            eloop_2 = eloop_2_
            eloop_3 = eloop_3_
            eloop_4 = eloop_4_
            eloop_5 = eloop_5_
            jloop_1 = jloop_1_
            
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
        eloop_1 = np.ones([batch_size, 60, 80, 96])
        eloop_2 = np.ones([batch_size, 15, 20, 256])
        eloop_3 = np.ones([batch_size, 15, 20, 384])
        eloop_4 = np.ones([batch_size, 15, 20, 384])
        eloop_5 = np.ones([batch_size, 15, 20, 256])
        jloop_1 = np.ones([batch_size, 60, 80, 96])
        
        loss_per_video = 0
        
        for image in image_list:
            # Get one image frame and pass it to network
            image_path = os.path.join(temp_folder, image)
            img = cv2.imread(image_path)
            
            eloop_1_, eloop_2_, eloop_3_, eloop_4_, eloop_5_, jloop_1_, cost, output = sess.run([
                                            network['conv1_in'],
                                            network['conv2_in'],
                                            network['conv3_in'],
                                            network['conv4_in'],
                                            network['conv5_in'],
                                            network['deconv5_in'],
                                            network['cost'],
                                            network['output']],
                                        feed_dict={network['image_input']: np.reshape(img, [batch_size, 240, 320, 3]),
                                                   network['conv1_eloop_input']: eloop_1,
                                                   network['conv2_eloop_input']: eloop_2,
                                                   network['conv3_eloop_input']: eloop_3,
                                                   network['conv4_eloop_input']: eloop_4,
                                                   network['conv5_eloop_input']: eloop_5,
                                                   network['conv1_jloop_input']: jloop_1,
                                                   network['lr']: lr,
                                                   network['label']: curr_label})
            
            # Update context layers
            eloop_1 = eloop_1_
            eloop_2 = eloop_2_
            eloop_3 = eloop_3_
            eloop_4 = eloop_4_
            eloop_5 = eloop_5_
            jloop_1 = jloop_1_
            
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
        eloop_1 = np.ones([batch_size, 60, 80, 96])
        eloop_2 = np.ones([batch_size, 15, 20, 256])
        eloop_3 = np.ones([batch_size, 15, 20, 384])
        eloop_4 = np.ones([batch_size, 15, 20, 384])
        eloop_5 = np.ones([batch_size, 15, 20, 256])
        jloop_1 = np.ones([batch_size, 60, 80, 96])
        
        loss_per_video = 0
        
        for image in image_list:
            # Get one image frame and pass it to network
            image_path = os.path.join(temp_folder, image)
            img = cv2.imread(image_path)
            
            eloop_1_, eloop_2_, eloop_3_, eloop_4_, eloop_5_, jloop_1_, cost, output = sess.run([
                                            network['conv1_in'],
                                            network['conv2_in'],
                                            network['conv3_in'],
                                            network['conv4_in'],
                                            network['conv5_in'],
                                            network['deconv5_in'],
                                            network['cost'],
                                            network['output']],
                                        feed_dict={network['image_input']: np.reshape(img, [batch_size, 240, 320, 3]),
                                                   network['conv1_eloop_input']: eloop_1,
                                                   network['conv2_eloop_input']: eloop_2,
                                                   network['conv3_eloop_input']: eloop_3,
                                                   network['conv4_eloop_input']: eloop_4,
                                                   network['conv5_eloop_input']: eloop_5,
                                                   network['conv1_jloop_input']: jloop_1,
                                                   network['lr']: lr,
                                                   network['label']: curr_label})
            
            # Update context layers
            eloop_1 = eloop_1_
            eloop_2 = eloop_2_
            eloop_3 = eloop_3_
            eloop_4 = eloop_4_
            eloop_5 = eloop_5_
            jloop_1 = jloop_1_
            
            loss_per_video += cost

            # Write image to folder            
#            cv2.imwrite(image_folder + '/' + image, deconv_out[0,:,:,0])
            
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
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            
         