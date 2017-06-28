import numpy as np
import tensorflow as tf

image_size = 96
nb_chanels = 1
nb_classes = 46
init_learning_rate = 0.05
#init_weight = 0.001

# 
def set_weight_conv_variable(shape, name):
	initial = tf.truncated_normal(shape=shape,stddev=0.001, mean = 0.0)
	return tf.Variable(initial,name=name)

def set_weight_fc_variable(shape, name):
	initial = tf.truncated_normal(shape=shape, stddev=0.001, mean = 0.0)
	return tf.Variable(initial_fc, name=name)

def set_bias_variable(shape, name):
	initial = tf.constant(0.0, shape = shape)
	return tf.Variable(initial,name=name)

def flatten(x):
	dim = x.get_shape().as_list()
	dim = np.prod(dim[1:])
	#dime = dim
	x = tf.reshape(x, [-1, dim])
	return x, dim

def conv2d(x, W, stride, padding_type='SAME'):
	return tf.nn.conv2d(x, W, [1,stride,stride, 1], padding_type)

def max_pool(x, filter, stride, padding_type='SAME'):
	return tf.nn.max_pool(x, [1,filter,filter,1], [1,stride,stride,1], padding_type)

def input() :
	x = tf.placeholder(tf.float32, [None, image_size, image_size, nb_chanels])
	y_label = tf.placeholder(tf.float32,[None, nb_classes])

	tf.add_to_collection('x',x)
	tf.add_to_collection('y_label',y_label)
	return x, y_label

def inference(x):

	x_image = tf.reshape(x, [-1,image_size,image_size,nb_chanels])
	#print("x_image : " + str(x_image.get_shape()))
	##print x_image.get_shape()
	#weights = []

	#Block 1
	depth1 = 32
	with tf.variable_scope('conv1') as scope:
		W_conv1 = set_weight_conv_variable([3,3,nb_chanels,depth1],'w')
		#weights.append(W_conv1)
		b_conv1 = set_bias_variable([depth1],'b')
		h_conv1 = conv2d(x_image,W_conv1,1) + b_conv1
		h_conv1 = tf.nn.relu(h_conv1, name=scope.name)

	#print("h_conv1 : "+str(h_conv1.get_shape()))
	h_pool1 = max_pool(h_conv1,2,2)
	#print("h_pool1 : " + str(h_pool1.get_shape()))
	
	#Block 2
	depth2 = 64
	with tf.variable_scope('conv2') as scope:
		W_conv2 = set_weight_conv_variable([3,3,depth1, depth2],'w')
		#weights.append(W_conv2)
		b_conv2 = set_bias_variable([depth2],'b')
		h_conv2 = conv2d(h_pool1,W_conv2,1) + b_conv2
		h_conv2 = tf.nn.relu(h_conv2, name=scope.name)

	with tf.variable_scope('conv3') as scope:
		W_conv3 = set_weight_conv_variable([3,3,depth2, depth2],'w')
		#weights.append(W_conv2)
		b_conv3 = set_bias_variable([depth2],'b')
		h_conv3 = conv2d(h_conv2,W_conv3,1) + b_conv3
		h_conv3 = tf.nn.relu(h_conv3, name=scope.name)

	#print("h_conv2 : "+str(h_conv2.get_shape()))
	h_pool2 = max_pool(h_conv3,2,2)
	#print("h_pool2 : "+str(h_pool2.get_shape()))

	#Block 3
	depth3 = 128
	with tf.variable_scope('conv4') as scope:
		W_conv4 = set_weight_conv_variable([3,3,depth2,depth3],'w')
		#weights.append(W_conv3)
		b_conv4 = set_bias_variable([depth3],'b')
		h_conv4 = conv2d(h_pool2,W_conv4,1) + b_conv4
		h_conv4 = tf.nn.relu(h_conv4, name=scope.name)

	#print("h_conv3 : "+str(h_conv3.get_shape()))

	with tf.variable_scope('conv5') as scope:
		W_conv5 = set_weight_conv_variable([3,3,depth3,depth3],'w')
		#weights.append(W_conv4)
		b_conv5 = set_bias_variable([depth3],'b')
		h_conv5 = conv2d(h_conv4,W_conv5,1) + b_conv5
		h_conv5 = tf.nn.relu(h_conv5, name=scope.name)

	#print("h_conv4 : "+str(h_conv4.get_shape()))
	h_pool3 = max_pool(h_conv5,2,2)
	#print("h_pool3 : "+str(h_pool3.get_shape()))

	#Block 4 
	depth4 = 256
	with tf.variable_scope('conv6') as scope:
		W_conv6 = set_weight_conv_variable([3,3,depth3,depth4],'w')
		#weights.append(W_conv5)
		b_conv6 = set_bias_variable([depth4],'b')
		h_conv6 = conv2d(h_pool3,W_conv6,1) + b_conv6
		h_conv6 = tf.nn.relu(h_conv6, name=scope.name)

	#print("h_conv5 : "+str(h_conv5.get_shape()))

	with tf.variable_scope('conv7') as scope:
		W_conv7 = set_weight_conv_variable([3,3,depth4,depth4],'w')
		#weights.append(W_conv6)
		b_conv7 = set_bias_variable([depth4],'b')
		h_conv7 = conv2d(h_conv6,W_conv7,1) + b_conv7
		h_conv7 = tf.nn.relu(h_conv7, name=scope.name)

	#print("h_conv6 : "+str(h_conv6.get_shape()))
	h_pool4 = max_pool(h_conv7,2,2)
	#print("h_pool4 : "+str(h_pool4.get_shape()))
	
	
	#Block 5
	depth5 = 512
	with tf.variable_scope('conv8') as scope:
		W_conv8 = set_weight_conv_variable([3,3,depth4,depth5],'w')
		#weights.append(W_conv7)
		b_conv8 = set_bias_variable([depth5],'b')
		h_conv8 = conv2d(h_pool4,W_conv8,1) + b_conv8
		h_conv8 = tf.nn.relu(h_conv8, name=scope.name)
	with tf.variable_scope('conv9') as scope:
		W_conv9 = set_weight_conv_variable([3,3,depth5,depth5],'w')
		#weights.append(W_conv7)
		b_conv9 = set_bias_variable([depth5],'b')
		h_conv9 = conv2d(h_conv8,W_conv9,1) + b_conv9
		h_conv9 = tf.nn.relu(h_conv9, name=scope.name)

	##print("h_conv7 : "+str(h_conv7.get_shape()))
	h_pool5 = max_pool(h_conv9,2,2)
	##print("h_pool5 : "+str(h_pool5.get_shape()))
	
	#Fully-connected layer 1
	h_pool_flat, size_weight = flatten(h_pool5)
	#print("h_pool_flat : "+str(h_pool_flat)+"  size_weight : "+str(size_weight))
	
	nb_neuron_fc1 = 1024
	with tf.variable_scope('fc1') as scope:
		w_fc1 = set_weight_fc_variable([size_weight, nb_neuron_fc1],'w')
		#weights.append(w_fc1)
		b_fc1 = set_bias_variable([nb_neuron_fc1],'b')
		h_fc1 = tf.matmul(h_pool_flat,w_fc1) + b_fc1
		h_fc1 = tf.nn.relu(h_fc1,name = scope.name)
		keep_prob_fc1 = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob_fc1)

	with tf.variable_scope('fc2') as scope:
		w_fc2 = set_weight_fc_variable([nb_neuron_fc1, nb_neuron_fc1],'w')
		#weights.append(w_fc1)
		b_fc2 = set_bias_variable([nb_neuron_fc1],'b')
		h_fc2 = tf.matmul(h_fc1_drop,w_fc2) + b_fc2
		h_fc2 = tf.nn.relu(h_fc2,name = scope.name)
		keep_prob_fc2 = tf.placeholder(tf.float32)
		h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob_fc2)

	#print("h_fc1_drop : "+str(h_fc1_drop.get_shape()))
	
	#Softmax Layer
	with tf.variable_scope('softmax') as scope:
		w_softmax = set_weight_fc_variable([nb_neuron_fc1,nb_classes],'w')
		#weights.append(w_softmax)
		b_softmax = set_bias_variable([nb_classes],'b')
		y_inference = tf.matmul(h_fc2_drop,w_softmax)+b_softmax

	#print("y_inference : "+str(y_inference.get_shape()))
	'''
	regul_loss = tf.constant(0.0)
	
	for i in range(len(weights)):
		regul_loss = tf.add(regul_loss, tf.nn.l2_loss(weights[i]))
	regul_loss = regul_loss*lamb
	'''

	tf.add_to_collection('keep_prob_fc1', keep_prob_fc1)
	tf.add_to_collection('keep_prob_fc2', keep_prob_fc2)
	tf.add_to_collection('y_inference', y_inference)
	#tf.add_to_collection('regul_loss', regul_loss)
	return y_inference

def loss(y_inference, y_label):
	#regul_loss = tf.get_collection('regul_loss')[0]
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_label, logits = y_inference))
	#total_loss = cross_entropy + regul_loss

	#tf.add_to_collection("origin_loss", cross_entropy)
	tf.add_to_collection('total_loss', cross_entropy)
	return cross_entropy

def train_op(loss, global_step):
	learning_rate = tf.train.exponential_decay(init_learning_rate,global_step,100000,0.96,staircase=False)
	#learning_rate = init_learning_rate
	train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss,global_step)

	tf.add_to_collection('train_step', train_step)
	tf.add_to_collection('learning_rate', learning_rate)

	return train_step
