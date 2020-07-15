# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:41:22 2020

@author: Prateek
"""

import tensorflow as tf

def Network(image_shape=[None, 28, 28, 1], lr=None, lr_beta=None):
    
    random_seed = 42
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        
        lr = tf.placeholder(tf.float32, [], name='Learning_Rate')
        image_input = tf.placeholder(tf.float32, image_shape, name='input_image')
        
        # *****************************************First Convolutional Layer*********************************************************
        k_h = 9; k_w = 9; channels = 1; filters = 1; s_h = 1; s_w = 1
        conv1w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv1w') #/ 100.0
        conv1b = tf.Variable(tf.truncated_normal([filters]), name='conv1b')
        conv1_in = tf.nn.conv2d(image_input, conv1w, [1, s_h, s_w, 1], padding='SAME') + conv1b
        
        # Elman Layer for the first convolutional layer
        conv1_eloop_input = tf.placeholder(tf.float32, conv1_in.get_shape().as_list(), name='conv1_eloop_input')
        conv1w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv1w_eloop') #/ 100.0
        conv1b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv1b_eloop')
        conv1_eloop_in = tf.nn.conv2d(conv1_eloop_input, conv1w_eloop, [1, 1, 1, 1], padding='SAME') + conv1b_eloop
        
        # Final Output from first convolutional layers
        conv1_output = tf.add(conv1_in, conv1_eloop_in)
        
        # Local response normalization for first convolutional layer 
#        radius = 2; bias = 1; alpha = 2e-5; beta = 0.75
#        lrn1 = tf.nn.local_response_normalization(conv1_output, depth_radius=radius, bias=bias, alpha=alpha, beta=beta)
        
        conv1_final_out = tf.nn.relu(conv1_output)
        
        print('End of Conv Layer 1', conv1_final_out.get_shape().as_list())
        # *****************************************First Convolutional Layer*********************************************************
        
        # *****************************************Second Convolutional Layer********************************************************
        k_h = 9; k_w = 9; s_h = 1; s_w = 1; channels = 1; filters = 1
        conv2w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv2w') #/ 100.0
        conv2b = tf.Variable(tf.truncated_normal([filters]), name='conv2b')
        conv2_in = tf.nn.conv2d(conv1_final_out, conv2w, [1, s_h, s_w, 1], padding='SAME') + conv2b
        
        # Elman Layer for second convolutional layer
        conv2_eloop_input = tf.placeholder(tf.float32, conv2_in.get_shape().as_list(), name='conv2_eloop_input')
        conv2w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv2w_eloop') #/ 100.0
        conv2b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv2b_eloop')
        conv2_eloop_in = tf.nn.conv2d(conv2_eloop_input, conv2w_eloop, [1, 1, 1, 1], padding='SAME') + conv2b_eloop
        
        # Final output from the second convolutional layers
        conv2_output = tf.add(conv2_in, conv2_eloop_in)
        
        # Local response normalization for second convolutional layer
#        radius = 2; bias = 2; alpha = 1e-4; beta = 0.75
#        lrn2 = tf.nn.local_response_normalization(conv2_output, depth_radius=radius, bias=bias, alpha=alpha, beta=beta)
        
        conv2_final_out = tf.nn.relu(conv2_output)
        
        print('End of Conv Layer 2', conv2_final_out.get_shape().as_list())        
        # *****************************************Second Convolutional Layer********************************************************
        
        # *****************************************First Fully Connected Layer*******************************************************
        fc_size_1 = 128; keep_prob = 0.75
        conv_out = tf.layers.flatten(conv2_final_out)
        print('Flatten', conv_out.get_shape().as_list())
        
        fc1w = tf.Variable(tf.truncated_normal([conv_out.get_shape().as_list()[1], fc_size_1]), name='fc1w')
        fc1b = tf.Variable(tf.truncated_normal([fc_size_1]), name='fc1b')
        fc1_in = tf.matmul(conv_out, fc1w) + fc1b
        fc1_out = tf.nn.dropout(fc1_in, keep_prob=keep_prob)
        
        print('End of FC 1', fc1_out.get_shape().as_list())
        # *****************************************First Fully Connected Layer*******************************************************
        
        # *****************************************Last Fully Connected Layer********************************************************
        classes = 4
        fc3w = tf.Variable(tf.truncated_normal([fc_size_1, classes]), name='fc3w')
        fc3b = tf.Variable(tf.truncated_normal([classes]), name='fc3b')
        fc3_in = tf.matmul(fc1_out, fc3w) + fc3b
        final_output = tf.nn.softmax(fc3_in)
        
        print('End of Softmax', final_output.get_shape().as_list())
        
        y_true = tf.placeholder(tf.float32, [1, classes], name='y_true')
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_in, labels=y_true))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
        
    return {'image_input': image_input,
            'output': final_output,
            'label': y_true,
            'cost': cost,
            'optimizer': optimizer,
            'lr': lr,
            'lr_beta': lr_beta,
            'conv1_in': conv1_in,
            'conv2_in': conv2_in,
            'conv1_eloop_input': conv1_eloop_input,
            'conv2_eloop_input': conv2_eloop_input,
            'conv1w': conv1w,
            'conv2w': conv2w,
            'conv1_final_out': conv1_final_out,
            'conv2_final_out': conv2_final_out}
    