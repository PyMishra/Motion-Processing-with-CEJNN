# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:28:08 2019

@author: Prateek
"""

import tensorflow as tf

def Network(image_shape=[None, 224, 224, 3], lr=None):
    
    random_seed = 42
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        
        image_input = tf.placeholder(tf.float32, image_shape, name='image_input')
#        loss_per_batch = tf.placeholder(tf.float32, [])
        lr = tf.placeholder(tf.float32, [], name='lr')
        
        # *****************************************First Convolutional Block*********************************************************
        k_h = 3; k_w = 3; channels = 3; filters = 64; s_h = 1; s_w = 1
        
        # First convolutional layer
        conv11w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv11w') / 100.0
        conv11b = tf.Variable(tf.truncated_normal([filters]), name='conv11b')
        conv11_in = tf.nn.conv2d(image_input, conv11w, [1, s_h, s_w, 1], padding='SAME') + conv11b
        
        # First Elman layer
        conv11_eloop_input = tf.placeholder(tf.float32, conv11_in.get_shape().as_list(), name='conv11_eloop_input')
        conv11w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv11w_eloop') / 100.0
        conv11b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv11b_eloop')
        conv11_eloop_in = tf.nn.conv2d(conv11_eloop_input, conv11w_eloop, [1, 1, 1, 1], padding='SAME') + conv11b_eloop
        
        # First Jordan layer
        conv11_jloop_input = tf.placeholder(tf.float32, conv11_in.get_shape().as_list(), name='conv11_jloop_input')
        conv11w_jloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv11w_jloop') / 100.0
        conv11b_jloop = tf.Variable(tf.truncated_normal([filters]), name='conv11b_jloop')
        conv11_jloop_in = tf.nn.conv2d(conv11_jloop_input, conv11w_jloop, [1, 1, 1, 1], padding='SAME') + conv11b_jloop
        
        # Final Output from first convolutional layer
        conv11_output = tf.add(tf.add(conv11_in, conv11_eloop_in), conv11_jloop_in)
        conv11_final_out = tf.nn.relu(conv11_output)
        
        
        # Second convolutional layer
        channels = 64
        conv12w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv12w') / 100.0
        conv12b = tf.Variable(tf.truncated_normal([filters]), name='conv12b')
        conv12_in = tf.nn.conv2d(conv11_final_out, conv12w, [1, s_h, s_w, 1], padding='SAME') + conv12b
        
        # Second Elman layer
        conv12_eloop_input = tf.placeholder(tf.float32, conv12_in.get_shape().as_list(), name='conv12_eloop_input')
        conv12w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv12w_eloop') / 100.0
        conv12b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv12b_eloop')
        conv12_eloop_in = tf.nn.conv2d(conv12_eloop_input, conv12w_eloop, [1, 1, 1, 1], padding='SAME') + conv12b_eloop
        
        # Final output from second convolutional layer
        conv12_output = tf.add(conv12_in, conv12_eloop_in)
        conv12_final_out = tf.nn.relu(conv12_output)
        
        max_pool_1 = tf.nn.max_pool(conv12_final_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        print('End of 1st conv block\nShape before max pool:', conv12_final_out.get_shape().as_list())
        print('Shape after max pool:', max_pool_1.get_shape().as_list())
        # *****************************************First Convolutional Block*********************************************************
        
        # ****************************************Second Convolutional Block*********************************************************
        k_h = 3; k_w = 3; channels = 64; filters = 128; s_h = 1; s_w = 1
        
        # First convolutional layer
        conv21w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv21w') / 100.0
        conv21b = tf.Variable(tf.truncated_normal([filters]), name='conv21b')
        conv21_in = tf.nn.conv2d(max_pool_1, conv21w, [1, s_h, s_w, 1], padding='SAME') + conv21b
        
        # First Elman layer
        conv21_eloop_input = tf.placeholder(tf.float32, conv21_in.get_shape().as_list(), name='conv21_eloop_input')
        conv21w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv21w_eloop') / 100.0
        conv21b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv21b_eloop')
        conv21_eloop_in = tf.nn.conv2d(conv21_eloop_input, conv21w_eloop, [1, 1, 1, 1], padding='SAME') + conv21b_eloop
        
        # Final output from first convolutional layer
        conv21_output = tf.add(conv21_in, conv21_eloop_in)
        conv21_final_out = tf.nn.relu(conv21_output)
        
        # Second convolutional layer
        channels = 128
        conv22w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv22w') / 100.0
        conv22b = tf.Variable(tf.truncated_normal([filters]), name='conv22b')
        conv22_in = tf.nn.conv2d(conv21_final_out, conv22w, [1, s_h, s_w, 1], padding='SAME') + conv22b
        
        # Second Elman layer
        conv22_eloop_input = tf.placeholder(tf.float32, conv22_in.get_shape().as_list(), name='conv22_eloop_input')
        conv22w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv22w_eloop') / 100.0
        conv22b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv22b_eloop')
        conv22_eloop_in = tf.nn.conv2d(conv22_eloop_input, conv22w_eloop, [1, 1, 1, 1], padding='SAME') + conv22b_eloop
        
        # Final output from second convolutional layer
        conv22_output = tf.add(conv22_in, conv22_eloop_in)
        conv22_final_out = tf.nn.relu(conv22_output)
        
        max_pool_2 = tf.nn.max_pool(conv22_final_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        print('End of 2nd conv block\nShape before max pool:', conv22_final_out.get_shape().as_list())
        print('Shape after max pool:', max_pool_2.get_shape().as_list())
        # ****************************************Second Convolutional Block*********************************************************
        
        # ****************************************Third Convolutional Block**********************************************************
        k_h = 3; k_w = 3; channels = 128; filters = 256; s_h = 1; s_w = 1
        
        # First convolutional layer
        conv31w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv31w') / 100.0
        conv31b = tf.Variable(tf.truncated_normal([filters]), name='conv31b')
        conv31_in = tf.nn.conv2d(max_pool_2, conv31w, [1, s_h, s_w, 1], padding='SAME') + conv31b
        
        # First Elman layer
        conv31_eloop_input = tf.placeholder(tf.float32, conv31_in.get_shape().as_list(), name='conv31_eloop_input')
        conv31w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv31w_eloop') / 100.0
        conv31b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv31b_eloop')
        conv31_eloop_in = tf.nn.conv2d(conv31_eloop_input, conv31w_eloop, [1, 1, 1, 1], padding='SAME') + conv31b_eloop
        
        # Final output from first convolutional layer
        conv31_output = tf.add(conv31_in, conv31_eloop_in)
        conv31_final_out = tf.nn.relu(conv31_output)
        
        # Second convolutional layer
        channels = 256
        conv32w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv32w') / 100.0
        conv32b = tf.Variable(tf.truncated_normal([filters]), name='conv32b')
        conv32_in = tf.nn.conv2d(conv31_final_out, conv32w, [1, s_h, s_w, 1], padding='SAME') + conv32b
        
        # Second Elman layer
        conv32_eloop_input = tf.placeholder(tf.float32, conv32_in.get_shape().as_list(), name='conv32_eloop_input')
        conv32w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv32w_eloop') / 100.0
        conv32b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv32b_eloop')
        conv32_eloop_in = tf.nn.conv2d(conv32_eloop_input, conv32w_eloop, [1, 1, 1, 1], padding='SAME') + conv32b_eloop
        
        # Final output from second convolutional layer
        conv32_output = tf.add(conv32_in, conv32_eloop_in)
        conv32_final_out = tf.nn.relu(conv32_output)
        
        max_pool_3 = tf.nn.max_pool(conv32_final_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        print('End of 3rd conv block\nShape before max pool:', conv32_final_out.get_shape().as_list())
        print('Shape after max pool:', max_pool_3.get_shape().as_list())
        # ****************************************Third Convolutional Block**********************************************************
        
        # ****************************************Fourth Convolutional Block*********************************************************
        k_h = 3; k_w = 3; channels = 256; filters = 512; s_h = 1; s_w = 1
        
        # First convolutional layer
        conv41w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv41w') / 100.0
        conv41b = tf.Variable(tf.truncated_normal([filters]), name='conv41b')
        conv41_in = tf.nn.conv2d(max_pool_3, conv41w, [1, s_h, s_w, 1], padding='SAME') + conv41b
        
        # First Elman layer
        conv41_eloop_input = tf.placeholder(tf.float32, conv41_in.get_shape().as_list(), name='conv41_eloop_input')
        conv41w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv41w_eloop') / 100.0
        conv41b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv41b_eloop')
        conv41_eloop_in = tf.nn.conv2d(conv41_eloop_input, conv41w_eloop, [1, 1, 1, 1], padding='SAME') + conv41b_eloop
        
        # Final output from first convolutional layer
        conv41_output = tf.add(conv41_in, conv41_eloop_in)
        conv41_final_out = tf.nn.relu(conv41_output)
        
        # Second convolutional layer
        channels = 512
        conv42w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv42w') / 100.0
        conv42b = tf.Variable(tf.truncated_normal([filters]), name='conv42b')
        conv42_in = tf.nn.conv2d(conv41_final_out, conv42w, [1, s_h, s_w, 1], padding='SAME') + conv42b
        
        # Second Elman layer
        conv42_eloop_input = tf.placeholder(tf.float32, conv42_in.get_shape().as_list(), name='conv42_eloop_input')
        conv42w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv42w_eloop') / 100.0
        conv42b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv42b_eloop')
        conv42_eloop_in = tf.nn.conv2d(conv42_eloop_input, conv42w_eloop, [1, 1, 1, 1], padding='SAME') + conv42b_eloop
        
        # Final output from second convolutional layer
        conv42_output = tf.add(conv42_in, conv42_eloop_in)
        conv42_final_out = tf.nn.relu(conv42_output)
        
        max_pool_4 = tf.nn.max_pool(conv42_final_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        print('End of 4th conv block\nShape before max pool:', conv42_final_out.get_shape().as_list())
        print('Shape after max pool:', max_pool_4.get_shape().as_list())
        # ****************************************Fourth Convolutional Block*********************************************************
        
        # ****************************************Fifth Convolutional Block**********************************************************
        k_h = 3; k_w = 3; channels = 512; filters = 512; s_h = 1; s_w = 1
        
        # First convolutional layer
        conv51w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv51w') / 100.0
        conv51b = tf.Variable(tf.truncated_normal([filters]), name='conv51b')
        conv51_in = tf.nn.conv2d(max_pool_4, conv51w, [1, s_h, s_w, 1], padding='SAME') + conv51b
        
        # First Elman layer
        conv51_eloop_input = tf.placeholder(tf.float32, conv51_in.get_shape().as_list(), name='conv51_eloop_input')
        conv51w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv51w_eloop') / 100.0
        conv51b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv51b_eloop')
        conv51_eloop_in = tf.nn.conv2d(conv51_eloop_input, conv51w_eloop, [1, 1, 1, 1], padding='SAME') + conv51b_eloop
        
        # Final output from first convolutional layer
        conv51_output = tf.add(conv51_in, conv51_eloop_in)
        conv51_final_out = tf.nn.relu(conv51_output)
        
        # Second convolutional layer
        channels = 512
        conv52w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv52w') / 100.0
        conv52b = tf.Variable(tf.truncated_normal([filters]), name='conv52b')
        conv52_in = tf.nn.conv2d(conv51_final_out, conv52w, [1, s_h, s_w, 1], padding='SAME') + conv52b
        
        # Second Elman layer
        conv52_eloop_input = tf.placeholder(tf.float32, conv52_in.get_shape().as_list(), name='conv52_eloop_input')
        conv52w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv52w_eloop') / 100.0
        conv52b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv52b_eloop')
        conv52_eloop_in = tf.nn.conv2d(conv52_eloop_input, conv52w_eloop, [1, 1, 1, 1], padding='SAME') + conv52b_eloop
        
        # Final output from second convolutional layer
        conv52_output = tf.add(conv52_in, conv52_eloop_in)
        conv52_final_out = tf.nn.relu(conv52_output)
        
        max_pool_5 = tf.nn.max_pool(conv52_final_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        print('End of 5th conv block\nShape before max pool:', conv52_final_out.get_shape().as_list())
        print('Shape after max pool:', max_pool_5.get_shape().as_list())
        # ****************************************Fifth Convolutional Block**********************************************************
        
        # *****************************************Deconvolutional Layer for Jordan**************************************************
        k_h = 3; k_w = 3; s_h = 16; s_w = 16; channels = 512; filters = 64
        batch_size = tf.shape(conv52_output)[0]
        deconv_shape = tf.stack([batch_size, 224, 224, 64])
        deconv5_w = tf.Variable(tf.truncated_normal([k_h, k_w, filters, channels], dtype=tf.float32, seed=random_seed), name='deconv5_w') / 100.0
        deconv5_b = tf.Variable(tf.truncated_normal([filters]), name='deconv5_b')
        deconv5_in = tf.nn.conv2d_transpose(conv52_output, deconv5_w, deconv_shape, [1, s_h, s_w, 1], padding='SAME') + deconv5_b # Big change here relu vs pre relu
        
        print('Deconv layer jordan', deconv5_in.get_shape().as_list())
        # *****************************************Deconvolutional Layer for Jordan**************************************************
        
        # *****************************************First Fully Connected Layer*******************************************************
        fc_size = 4096; drop_out = 0.5
        conv_out = tf.layers.flatten(max_pool_5)
#        print(conv_out.get_shape().as_list())
        
        fc1w = tf.Variable(tf.truncated_normal([conv_out.get_shape().as_list()[1], fc_size]), name='fc1w')
        fc1b = tf.Variable(tf.truncated_normal([fc_size]), name='fc1b')
        fc1_in = tf.matmul(conv_out, fc1w) + fc1b
        fc1_out = tf.nn.dropout(fc1_in, keep_prob=drop_out)
        
        print('End of FC 1', fc1_out.get_shape().as_list())
        # *****************************************First Fully Connected Layer*******************************************************
        
        # *****************************************Second Fully Connected Layer******************************************************
        fc2w = tf.Variable(tf.truncated_normal([fc_size, fc_size]), name='fc2w')
        fc2b = tf.Variable(tf.truncated_normal([fc_size]), name='fc2b')
        fc2_in = tf.matmul(fc1_out, fc2w) + fc2b
        fc2_out = tf.nn.dropout(fc2_in, keep_prob=drop_out)
        
        print('End of FC 2', fc2_out.get_shape().as_list())
        # *****************************************Second Fully Connected Layer******************************************************
        
        # *****************************************Final Softmax Layer***************************************************************
        classes = 11
        fc3w = tf.Variable(tf.truncated_normal([fc_size, classes]), name='fc3w')
        fc3b = tf.Variable(tf.truncated_normal([classes]), name='fc3b')
        fc3_in = tf.matmul(fc2_out, fc3w) + fc3b
        final_output = tf.nn.softmax(fc3_in)
        
        print('End of Softmax', final_output.get_shape().as_list())
        # *****************************************Final Softmax Layer***************************************************************
        
        y_true = tf.placeholder(tf.float32, [1, 11], name='y_true')
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_in, labels=y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)        
    
        
    return {'image_input': image_input,
            'output': final_output,
            'label': y_true,
            'cost': cost,
            'optimizer': optimizer,
            'lr': lr,
            'conv11_in': conv11_in,
            'conv12_in': conv12_in,
            'conv21_in': conv21_in,
            'conv22_in': conv22_in,
            'conv31_in': conv31_in,
            'conv32_in': conv32_in,
            'conv41_in': conv41_in,
            'conv42_in': conv42_in,
            'conv51_in': conv51_in,
            'conv52_in': conv52_in,
            'deconv5_in': deconv5_in,
            'conv11_eloop_input': conv11_eloop_input,
            'conv12_eloop_input': conv12_eloop_input,
            'conv21_eloop_input': conv21_eloop_input,
            'conv22_eloop_input': conv22_eloop_input,
            'conv31_eloop_input': conv31_eloop_input,
            'conv32_eloop_input': conv32_eloop_input,
            'conv41_eloop_input': conv41_eloop_input,
            'conv42_eloop_input': conv42_eloop_input,
            'conv51_eloop_input': conv51_eloop_input,
            'conv52_eloop_input': conv52_eloop_input,
            'conv11_jloop_input': conv11_jloop_input,
            }
        
        
        
    