# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:19:32 2019

@author: Prateek
"""

import tensorflow as tf

def Network(image_shape=[None, 240, 320, 3], lr=None):
    
    random_seed = 42
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        
        image_input = tf.placeholder(tf.float32, image_shape, name='image_input')
#        loss_per_batch = tf.placeholder(tf.float32, [])
        lr = tf.placeholder(tf.float32, [], name='lr')
        
        # *****************************************First Convolutional Layer*********************************************************
        k_h = 11; k_w = 11; channels = 3; filters = 96; s_h = 4; s_w = 4
        conv1w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv1w') / 100.0
        conv1b = tf.Variable(tf.truncated_normal([filters]), name='conv1b')
        conv1_in = tf.nn.conv2d(image_input, conv1w, [1, s_h, s_w, 1], padding='SAME') + conv1b
        
        # Elman Layer for the first convolutional layer
        conv1_eloop_input = tf.placeholder(tf.float32, conv1_in.get_shape().as_list(), name='conv1_eloop_input')
        conv1w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv1w_eloop') / 100.0
        conv1b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv1b_eloop')
        conv1_eloop_in = tf.nn.conv2d(conv1_eloop_input, conv1w_eloop, [1, 1, 1, 1], padding='SAME') + conv1b_eloop
        
        # Jordan Layer for the first convolutional layer
        conv1_jloop_input = tf.placeholder(tf.float32, conv1_in.get_shape().as_list(), name='conv1_jloop_input')
        conv1w_jloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv1w_jloop') / 100.0
        conv1b_jloop = tf.Variable(tf.truncated_normal([filters]), name='conv1b_jloop')
        conv1_jloop_in = tf.nn.conv2d(conv1_jloop_input, conv1w_jloop, [1, 1, 1, 1], padding='SAME') + conv1b_jloop
        
        # Final Output from first convolutional layers
        conv1_output = tf.add(tf.add(conv1_in, conv1_eloop_in), conv1_jloop_in)
        conv1_final_out = tf.nn.relu(conv1_output)
        
        # Local response normalization for first convolutional layer 
        radius = 2; bias = 2; alpha = 1e-4; beta = 0.75
        lrn1 = tf.nn.local_response_normalization(conv1_final_out, depth_radius=radius, bias=bias, alpha=alpha, beta=beta)
        
        # Max pooling
#        max_pool_1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
#        print('Eloop_1 shape:', conv1_in.get_shape().as_list())
        print('End of Conv Layer 1', lrn1.get_shape().as_list())        
        # *****************************************First Convolutional Layer*********************************************************
        
        # *****************************************Second Convolutional Layer********************************************************
        k_h = 5; k_w = 5; s_h = 4; s_w = 4; channels = 96; filters = 256
        conv2w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv2w') / 100.0
        conv2b = tf.Variable(tf.truncated_normal([filters]), name='conv2b')
        conv2_in = tf.nn.conv2d(lrn1, conv2w, [1, s_h, s_w, 1], padding='SAME') + conv2b
        
        # Elman Layer for second convolutional layer
        conv2_eloop_input = tf.placeholder(tf.float32, conv2_in.get_shape().as_list(), name='conv2_eloop_input')
        conv2w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv2w_eloop') / 100.0
        conv2b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv2b_eloop')
        conv2_eloop_in = tf.nn.conv2d(conv2_eloop_input, conv2w_eloop, [1, 1, 1, 1], padding='SAME') + conv2b_eloop
        
        # Final output from the second convolutional layers
        conv2_output = tf.add(conv2_in, conv2_eloop_in)
        conv2_final_out = tf.nn.relu(conv2_output)
        
        # Local response normalization for second convolutional layer
        radius = 2; bias = 2; alpha = 1e-4; beta = 0.75
        lrn2 = tf.nn.local_response_normalization(conv2_final_out, depth_radius=radius, bias=bias, alpha=alpha, beta=beta)
        
        # Max pooling
#        max_pool_2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
#        print('Eloop_2 shape:', conv2_in.get_shape().as_list())
        print('End of Conv Layer 2', lrn2.get_shape().as_list())        
        # *****************************************Second Convolutional Layer********************************************************
        
        # *****************************************Third Convolutional Layer*********************************************************
        k_h = 3; k_w = 3; s_h = 1; s_w = 1; channels = 256; filters = 384
        conv3w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv3w') / 100.0
        conv3b = tf.Variable(tf.truncated_normal([filters]), name='conv3b')
        conv3_in = tf.nn.conv2d(lrn2, conv3w, [1, s_h, s_w, 1], padding='SAME') + conv3b
        
        # Elman Layer for third convolutional layer
        conv3_eloop_input = tf.placeholder(tf.float32, conv3_in.get_shape().as_list(), name='conv3_eloop_input')
        conv3w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv3w_eloop') / 100.0
        conv3b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv3b_eloop')
        conv3_eloop_in = tf.nn.conv2d(conv3_eloop_input, conv3w_eloop, [1, 1, 1, 1], padding='SAME') + conv3b_eloop
        
        # Final output from the second convolutional layers
        conv3_output = tf.add(conv3_in, conv3_eloop_in)
        conv3_final_out = tf.nn.relu(conv3_output)
        
#        print('Eloop_3 shape:', conv3_in.get_shape().as_list())
        print('End of Conv Layer 3, no mp or lrn', conv3_final_out.get_shape().as_list())        
        # *****************************************Third Convolutional Layer*********************************************************
        
        # *****************************************Fourth Convolutional Layer********************************************************
        k_h = 3; k_w = 3; s_h = 1; s_w = 1; channels = 384; filters = 384
        conv4w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv4w') / 100.0
        conv4b = tf.Variable(tf.truncated_normal([filters]), name='conv4b')
        conv4_in = tf.nn.conv2d(conv3_final_out, conv4w, [1, s_h, s_w, 1], padding='SAME') + conv4b
        
        # Elman Layer for fourth convolutional layer
        conv4_eloop_input = tf.placeholder(tf.float32, conv4_in.get_shape().as_list(), name='conv4_eloop_input')
        conv4w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv4w_eloop') / 100.0
        conv4b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv4b_eloop')
        conv4_eloop_in = tf.nn.conv2d(conv4_eloop_input, conv4w_eloop, [1, 1, 1, 1], padding='SAME') + conv4b_eloop
        
        # Final output from the second convolutional layers
        conv4_output = tf.add(conv4_in, conv4_eloop_in)
        conv4_final_out = tf.nn.relu(conv4_output)
        
#        print('Eloop_4 shape:', conv4_in.get_shape().as_list())
        print('End of Conv Layer 4, no mp or lrn', conv4_final_out.get_shape().as_list())        
        # *****************************************Fourth Convolutional Layer********************************************************
        
        # *****************************************Fifth Convolutional Layer*********************************************************
        k_h = 3; k_w = 3; s_h = 1; s_w = 1; channels = 384; filters = 256
        conv5w = tf.Variable(tf.truncated_normal([k_h, k_w, channels, filters], dtype=tf.float32, seed=random_seed), name='conv5w') / 100.0
        conv5b = tf.Variable(tf.truncated_normal([filters]), name='conv5b')
        conv5_in = tf.nn.conv2d(conv4_final_out, conv5w, [1, s_h, s_w, 1], padding='SAME') + conv5b
        
        # Elman Layer for fifth convolutional layer
        conv5_eloop_input = tf.placeholder(tf.float32, conv5_in.get_shape().as_list(), name='conv5_eloop_input')
        conv5w_eloop = tf.Variable(tf.truncated_normal([k_h, k_w, filters, filters], dtype=tf.float32, seed=random_seed), name='conv5w_eloop') / 100.0
        conv5b_eloop = tf.Variable(tf.truncated_normal([filters]), name='conv5b_eloop')
        conv5_eloop_in = tf.nn.conv2d(conv5_eloop_input, conv5w_eloop, [1, 1, 1, 1], padding='SAME') + conv5b_eloop
        
        # Final output from the fifth convolutional layers
        conv5_output = tf.add(conv5_in, conv5_eloop_in)
        conv5_final_out = tf.nn.relu(conv5_output)
        
        # Max pooling for the fifth convolutional layer
        max_pool_5 = tf.nn.max_pool(conv5_final_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
#        print('Eloop_5 shape:', conv5_in.get_shape().as_list())
        print('End of Conv Layer 5, no lrn', conv5_final_out.get_shape().as_list())
        print('Max pool', max_pool_5.get_shape().as_list())
        # *****************************************Fifth Convolutional Layer*********************************************************
        
        # ***************************************** Deconvolutional Layer for output*************************************************
#        k_h = 1; k_w = 1; s_h = 4; s_w = 4; channels = 256; filters = 1
#        batch_size = tf.shape(conv5_output)[0]
#        deconv_shape = tf.stack([batch_size, 60, 80, 1])
#        deconv_w = tf.Variable(tf.truncated_normal([k_h, k_w, filters, channels], dtype=tf.float32, seed=random_seed), name='deconv_w') / 100.0
#        deconv_b = tf.Variable(tf.truncated_normal([filters]), name='deconv_b')
#        deconv_in = tf.nn.conv2d_transpose(conv5_final_out, deconv_w, deconv_shape, [1, s_h, s_w, 1], padding='SAME') + deconv_b
#        deconv_out = tf.nn.relu(deconv_in)
#        
#        print('Deconv layer out', deconv_in.get_shape().as_list())
#        print('Deconv out image', deconv_out.get_shape().as_list())
        # *****************************************Deconvolutional Layer for output**************************************************
        
        # *****************************************Deconvolutional Layer for Jordan**************************************************
        k_h = 3; k_w = 3; s_h = 4; s_w = 4; channels = 256; filters = 96
        batch_size = tf.shape(conv5_output)[0]
        deconv_shape = tf.stack([batch_size, 60, 80, 96])
        deconv5_w = tf.Variable(tf.truncated_normal([k_h, k_w, filters, channels], dtype=tf.float32, seed=random_seed), name='deconv5_w') / 100.0
        deconv5_b = tf.Variable(tf.truncated_normal([filters]), name='deconv5_b')
        deconv5_in = tf.nn.conv2d_transpose(conv5_output, deconv5_w, deconv_shape, [1, s_h, s_w, 1], padding='SAME') + deconv5_b
        
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
        
        y_true = tf.placeholder(tf.float32, [1, 11], name='y_true')
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_in, labels=y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)        
        
    return {'image_input': image_input,
            'output': final_output,
            'label': y_true,
            'cost': cost,
            'optimizer': optimizer,
            'lr': lr,
            'conv1_in': conv1_in,
            'conv2_in': conv2_in,
            'conv3_in': conv3_in,
            'conv4_in': conv4_in,
            'conv5_in': conv5_in,
#            'deconv_in': deconv_in,
#            'deconv_out': deconv_out,
            'conv_out': conv_out,
            'conv1_eloop_input': conv1_eloop_input,
            'conv2_eloop_input': conv2_eloop_input,
            'conv3_eloop_input': conv3_eloop_input,
            'conv4_eloop_input': conv4_eloop_input,
            'conv5_eloop_input': conv5_eloop_input,
            'conv1_jloop_input': conv1_jloop_input,
            'deconv5_in': deconv5_in
            }

            
            
            
            
            
    