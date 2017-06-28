import tensorflow as tf
import numpy as np

img_size = 96
started_learning_rate = 0.005

def new_weights(shape,name):
    initial = tf.truncated_normal(shape=shape, stddev=0.001, mean=0.0)
    return tf.Variable(initial, name=name)

def new_bias(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)

def new_conv2d(pre_layer, filter_size, nb_chanels_input, nb_filters, n_scope, relu=True):
    with tf.variable_scope(n_scope) as scope:
        shape = [filter_size,filter_size,nb_chanels_input,nb_filters]
        weight = new_weights(shape=shape, name='W')
        bias = new_bias(shape=[nb_filters], name='b')

        layer = tf.nn.conv2d(input=pre_layer, filter=weight, strides=[1,1,1,1], padding='SAME')

        layer += bias

        if relu :
            layer = tf.nn.relu(layer,name=scope.name)

        return layer, weight

def new_maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    nb_dim = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer,[-1,nb_dim])

    return layer_flat, nb_dim

def new_fc_layer(pre_layer, nb_inputs, nb_outputs, n_scope, relu = True):
    with tf.variable_scope(n_scope) as scope :
        weight = new_weights(shape=[nb_inputs,nb_outputs], name='W')
        bias = new_bias(shape=[nb_outputs], name='b')

        layer = tf.matmul(pre_layer,weight) + bias

        if relu :
            layer = tf.nn.relu(layer, name=scope.name)
     
    	return layer

def input():
    x = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
    y_ = tf.placeholder(tf.float32, [None, 46])

    tf.add_to_collection('x',x)
    tf.add_to_collection('y_',y_)

    return x, y_

def inference(x):
    x_image = tf.reshape(x,[-1, img_size, img_size, 1])
    print('x_image = '+str(x_image.get_shape()))

    conv1, W_conv1 = new_conv2d(pre_layer=x, filter_size=3, nb_chanels_input=1, nb_filters=32, relu=False, n_scope='conv1')
    print('conv1 = '+str(conv1.get_shape())+"   W_conv1 = "+str(W_conv1.get_shape()))
    conv1 = new_maxpool2d(conv1)
    print('conv1_pool = '+str(conv1.get_shape()))

    conv2, W_conv2 = new_conv2d(pre_layer=conv1, filter_size=3, nb_chanels_input=32, nb_filters=32, relu=False, n_scope='conv2')
    print('conv2 = '+str(conv2.get_shape())+"   W_conv2 = "+str(W_conv2.get_shape()))
    conv2 = new_maxpool2d(conv2)
    print('conv2_pool = '+str(conv2.get_shape()))

    conv3, W_conv3 = new_conv2d(pre_layer=conv2, filter_size=3, nb_chanels_input=32, nb_filters=64, relu=False, n_scope='conv3')
    print('conv3 = '+str(conv3.get_shape())+"   W_conv3 = "+str(W_conv3.get_shape()))
    conv3 = new_maxpool2d(conv3)
    print('conv3_pool = '+str(conv3.get_shape()))

    conv4, W_conv4 = new_conv2d(pre_layer=conv3, filter_size=3, nb_chanels_input=64, nb_filters=64, relu=False, n_scope='conv4')
    print('conv4 = '+str(conv4.get_shape())+"   W_conv4="+str(W_conv4.get_shape()))
    conv4 = new_maxpool2d(conv4)
    print('conv4_pool = '+str(conv4.get_shape()))
    
    conv5, W_conv5 = new_conv2d(pre_layer=conv4, filter_size=3, nb_chanels_input=64, nb_filters=128, relu=False, n_scope='conv5')
    print('conv5 = '+str(conv5.get_shape())+"   W_conv5="+str(W_conv5.get_shape()))
    conv5 = new_maxpool2d(conv5)
    print('conv5_pool = '+str(conv5.get_shape()))

    layer_flat, nb_dim = flatten_layer(conv5)
    print('layer_flat = '+str(layer_flat.get_shape())+'     nb_dim = '+str(nb_dim))

    fc1 = new_fc_layer(pre_layer=layer_flat, nb_inputs=nb_dim, nb_outputs= 4096, n_scope='fc1')
    print('fc1 = '+str(fc1.get_shape()))
    keep_prob_fc1 = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, keep_prob_fc1)
    print('fc1_drop = '+str(fc1_drop.get_shape()))

    fc2 = new_fc_layer(pre_layer=fc1_drop, nb_inputs=4096, nb_outputs=4096, n_scope='fc2')
    print('fc2 = '+str(fc2.get_shape()))
    keep_prob_fc2 = tf.placeholder(tf.float32)
    fc2_drop = tf.nn.dropout(fc2, keep_prob_fc2)
    print('fc2_drop = '+str(fc2_drop.get_shape()))
    y_conv = new_fc_layer(pre_layer=fc2_drop, nb_inputs=4096, nb_outputs=46, relu= False, n_scope='softmax')
    print('y_conv = '+str(y_conv.get_shape()))

    tf.add_to_collection('y_conv',y_conv)
    tf.add_to_collection('keep_prob_fc1',keep_prob_fc1)
    tf.add_to_collection('keep_prob_fc2',keep_prob_fc2)

    return y_conv

def loss(y_conv, y_):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    tf.add_to_collection('cross_entropy_loss',cross_entropy)
    
    return cross_entropy

def train_op(loss, global_step):
    learning_rate = tf.train.exponential_decay(started_learning_rate, global_step, 100000, 0.96, staircase=False)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss,global_step)

    tf.add_to_collection('learning_rate', learning_rate)
    tf.add_to_collection('train_step', train_step)

    return train_step