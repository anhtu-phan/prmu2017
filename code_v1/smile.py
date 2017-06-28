import numpy as np
import cv2
import tensorflow as tf


image_size = 96
nb_channels = 1
nb_classes = 46
lamb = 0.02
init_lr = 0.005
use_bn = False
bn_decay = 0.99
epsilon = 0.001

# Initialize weight variable use Glorot normal.
def weight_conv_variable(shape,name):
    std = shape[0] * shape[1] * shape[2]
    std = np.sqrt(2. / std)
    initial = tf.truncated_normal(shape, stddev=0.001, mean=0.0)
    return tf.Variable(initial,name=name)

def weight_fc_variable(shape,name):
    std = shape[0]
    std = np.sqrt(2. / std)
    initial = tf.truncated_normal(shape, stddev=0.001, mean=0.0)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial,name=name)

def flatten(x):
    dim = x.get_shape().as_list()
    dim = np.prod(dim[1:])
    dime = dim
    x = tf.reshape(x, [-1, dim])
    return x, dime

def conv2d(x, W, stride, padding_type = 'SAME'):
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], padding_type)

def max_pool(x, filter, stride, padding_type = 'SAME'):
    return tf.nn.max_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], padding_type)

def batch_normalization(type,input,is_training,decay,variable_averages):
    shape=np.shape(input)
    if type == 'conv':
        gamma=tf.Variable(tf.constant(1.,shape=[shape[3]]))
        beta=tf.Variable(tf.constant(0.,shape=[shape[3]]))
        batch_mean, batch_var = tf.nn.moments(input, [0,1,2])
        pop_mean= tf.Variable(tf.zeros([shape[3]],dtype=tf.float32),trainable=False)
        pop_var = tf.Variable(tf.ones([shape[3]], dtype=tf.float32), trainable=False)
    elif type == 'fc':
        gamma=tf.Variable(tf.constant(1.,shape=[shape[1]]))
        beta=tf.Variable(tf.constant(0.,shape=[shape[1]]))
        batch_mean,batch_var = tf.nn.moments(input, [0])
        pop_mean= tf.Variable(tf.zeros([shape[1]],dtype=tf.float32),trainable=False)
        pop_var = tf.Variable(tf.ones([shape[1]], dtype=tf.float32), trainable=False)


    def update_mean_var():
        update_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        update_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([update_mean,update_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)
    mean,var=tf.cond(is_training,update_mean_var,lambda: (pop_mean,pop_var))
    return tf.nn.batch_normalization(input,mean,var,beta,gamma,epsilon)


def input():
    x = tf.placeholder(tf.float32, [None, image_size, image_size, nb_channels])
    y_ = tf.placeholder(tf.float32, [None, nb_classes])

    tf.add_to_collection('x', x)
    tf.add_to_collection('y_', y_)
    return x,y_

def inference(x):
    #variable_averages = tf.train.ExponentialMovingAverage(bn_decay)
    #is_training = tf.placeholder(dtype=tf.bool)

    x_image = tf.reshape(x, [-1, image_size, image_size, nb_channels])

    # Block 1

    with tf.variable_scope('conv1') as scope:
        W_conv1 = weight_conv_variable([3,3,nb_channels,32],'W')
        b_conv1 = bias_variable([32],'b')
        h_conv1 = conv2d(x_image,W_conv1,1) + b_conv1
        if use_bn:
            h_conv1 = batch_normalization('conv',h_conv1,is_training,bn_decay,variable_averages)
        h_conv1 = tf.nn.relu(h_conv1,name= scope.name)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = weight_conv_variable([3,3,32,32],'W')
        b_conv2 = bias_variable([32],'b')
        h_conv2 = conv2d(h_conv1,W_conv2,1) + b_conv2
        if use_bn:
            h_conv2 = batch_normalization('conv',h_conv2,is_training,bn_decay,variable_averages)
        h_conv2 = tf.nn.relu(h_conv2,name= scope.name)

    h_pool1 = max_pool(h_conv2,2,2)

    # Block 2

    with tf.variable_scope('conv3') as scope:
        W_conv3 = weight_conv_variable([3,3,32,64],'W')
        b_conv3 = bias_variable([64],'b')
        h_conv3 = conv2d(h_pool1,W_conv3,1) + b_conv3
        if use_bn:
            h_conv3 = batch_normalization('conv',h_conv3,is_training,bn_decay,variable_averages)
        h_conv3 = tf.nn.relu(h_conv3,name= scope.name)

    with tf.variable_scope('conv4') as scope:
        W_conv4 = weight_conv_variable([3,3,64,64],'W')
        b_conv4 = bias_variable([64],'b')
        h_conv4 = conv2d(h_conv3,W_conv4,1) + b_conv4
        if use_bn:
            h_conv4 = batch_normalization('conv',h_conv4,is_training,bn_decay,variable_averages)
        h_conv4 = tf.nn.relu(h_conv4,name= scope.name)

    h_pool2 = max_pool(h_conv4,2,2)

    # Block 3

    with tf.variable_scope('conv5') as scope:
        W_conv5 = weight_conv_variable([3,3,64,128],'W')
        b_conv5 = bias_variable([128],'b')
        h_conv5 = conv2d(h_pool2,W_conv5,1) + b_conv5
        if use_bn:
            h_conv5 = batch_normalization('conv',h_conv5,is_training,bn_decay,variable_averages)
        h_conv5 = tf.nn.relu(h_conv5,name= scope.name)

    with tf.variable_scope('conv6') as scope:
        W_conv6 = weight_conv_variable([3,3,128,128],'W')
        b_conv6 = bias_variable([128],'b')
        h_conv6 = conv2d(h_conv5,W_conv6,1) + b_conv6
        if use_bn:
            h_conv6 = batch_normalization('conv',h_conv6,is_training,bn_decay,variable_averages)
        h_conv6 = tf.nn.relu(h_conv6,name= scope.name)

    h_pool3 = max_pool(h_conv6,2,2)

    # Block 4

    with tf.variable_scope('conv7') as scope:
        W_conv7 = weight_conv_variable([3,3,128,256],'W')
        b_conv7 = bias_variable([256],'b')
        h_conv7 = conv2d(h_pool3,W_conv7,1) + b_conv7
        if use_bn:
            h_conv7 = batch_normalization('conv',h_conv7,is_training,bn_decay,variable_averages)
        h_conv7 = tf.nn.relu(h_conv7,name= scope.name)

    with tf.variable_scope('conv8') as scope:
        W_conv8 = weight_conv_variable([3,3,256,256],'W')
        b_conv8 = bias_variable([256],'b')
        h_conv8 = conv2d(h_conv7,W_conv8,1) + b_conv8
        if use_bn:
            h_conv8 = batch_normalization('conv',h_conv8,is_training,bn_decay,variable_averages)
        h_conv8 = tf.nn.relu(h_conv8,name= scope.name)

    with tf.variable_scope('conv9') as scope:
        W_conv9 = weight_conv_variable([3,3,256,256],'W')
        b_conv9 = bias_variable([256],'b')
        h_conv9 = conv2d(h_conv8,W_conv9,1) + b_conv9
        if use_bn:
            h_conv9 = batch_normalization('conv',h_conv9,is_training,bn_decay,variable_averages)
        h_conv9 = tf.nn.relu(h_conv9,name= scope.name)

    h_pool4 = max_pool(h_conv9,2,2)

    # Dense Layer 1

    h_pool4_flat, size_weight = flatten(h_pool4)

    with tf.variable_scope('fc1') as scope:
        W_fc1 = weight_fc_variable([size_weight,256],'W')
        b_fc1 = bias_variable([256],'b')
        h_fc1 = tf.matmul(h_pool4_flat,W_fc1) + b_fc1
        if use_bn:
            h_fc1 = batch_normalization('fc',h_fc1,is_training,bn_decay,variable_averages)
        h_fc1 = tf.nn.relu(h_fc1,name = scope.name)
        keep_prob_fc1 = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc1)

    # Dense Layer 2

    with tf.variable_scope('fc2') as scope:
        W_fc2 = weight_fc_variable([256,256],'W')
        b_fc2 = bias_variable([256],'b')
        h_fc2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
        if use_bn:
            h_fc2 = batch_normalization('fc',h_fc2,is_training,bn_decay,variable_averages)
        h_fc2 = tf.nn.relu(h_fc2,name = scope.name)
        keep_prob_fc2 = tf.placeholder(tf.float32)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_fc2)

    # Softmax Layer

    with tf.variable_scope('softmax') as scope:
        W_fc3 = weight_fc_variable([256,46],'W')
        b_fc3 = bias_variable([46],'b')
        y_conv = tf.matmul(h_fc2_drop,W_fc3) + b_fc3

    #-------------------------------- END OF MODEL ---------------------------------------------#

    # Define summary op

    '''loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    acc_train_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('accuracy', acc_train_placeholder)
    for var in [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4,
                W_conv5, b_conv5, W_conv6, b_conv6, W_conv7, b_conv7, W_conv8, b_conv8,
                W_conv9, b_conv9, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3]:
        tf.summary.histogram(var.op.name, var)
    summary_op = tf.summary.merge_all()

    # L2-loss

    regul_loss = lamb * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss((W_conv3))
                         + tf.nn.l2_loss((W_conv4)) + tf.nn.l2_loss((W_conv5)) + tf.nn.l2_loss((W_conv6))
                         + tf.nn.l2_loss((W_conv7)) + tf.nn.l2_loss((W_conv8)) + tf.nn.l2_loss((W_conv9))
                         + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3))
    '''
    # Add to collection

    tf.add_to_collection('keep_prob_fc1', keep_prob_fc1)
    tf.add_to_collection('keep_prob_fc2', keep_prob_fc2)
    tf.add_to_collection('y_conv', y_conv)
    #tf.add_to_collection('regul_loss', regul_loss)
    #tf.add_to_collection('loss_summary_placeholder', loss_summary_placeholder)
    #tf.add_to_collection('acc_train_placeholder', acc_train_placeholder)
    #tf.add_to_collection('summary_op', summary_op)
    #tf.add_to_collection('is_training', is_training)

    return y_conv

def loss(y_conv,y_):
    #regul_loss = tf.get_collection('regul_loss')[0]
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    #total_loss = cross_entropy + regul_loss

    #tf.add_to_collection('origin_loss', cross_entropy)
    tf.add_to_collection('total_loss', cross_entropy)
    return cross_entropy

def define_additional_variables():
    current_epoch=tf.Variable(-1)
    current_step=tf.Variable(-1)
    max_acc_tensor=tf.Variable(0.,dtype=tf.float32)

    tf.add_to_collection('current_epoch', current_epoch)
    tf.add_to_collection('current_step',current_step)
    tf.add_to_collection('max_acc_tensor', max_acc_tensor)
    return  current_epoch,current_step,max_acc_tensor

def train_op(loss, global_step):
    learning_rate = tf.train.exponential_decay(init_lr,global_step,100000,0.96,staircase=False)

    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss,global_step)

    tf.add_to_collection('train_step', train_step)
    tf.add_to_collection('learning_rate', learning_rate)
    return train_step








