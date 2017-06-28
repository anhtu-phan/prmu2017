import model
import numpy as np
import os
import tensorflow as tf 
#import read_data
from read_dataset import PRMUDataSet

nb_epochs = 1000
batch_size = 32 
drop_out_prob = 0.5
#nb_images_train = 3000

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train():
	sess = tf.InteractiveSession()
	global_step = tf.contrib.framework.get_or_create_global_step()

	#Load data
	train_sets = PRMUDataSet("1_train_0.9")
	train_sets.load_data_target()
	n_train_samples = train_sets.get_n_types_target()
	print ("n_train_samples = "+str(n_train_samples))
	
	if not os.path.isfile('save/current/model.ckpt.index'):		
		print('Create new model')
		x, y_ = model.input()
		y_conv = model.inference(x)
		loss = model.loss(y_conv=y_conv, y_=y_)
		train_step = model.train_op(loss, global_step)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
	else:
		print('Load exist model')
		saver = tf.train.import_meta_graph('save/current/model.ckpt.meta')
		saver.restore(sess, 'save/current/model.ckpt')

	
	learning_rate = tf.get_collection('learning_rate')[0]
	cross_entropy_loss = tf.get_collection('cross_entropy_loss')[0]

	train_step = tf.get_collection('train_step')[0]

	keep_prob_fc1 = tf.get_collection('keep_prob_fc1')[0]
	keep_prob_fc2 = tf.get_collection('keep_prob_fc2')[0]
	x = tf.get_collection('x')[0]
	y_ = tf.get_collection('y_')[0]
	y_conv = tf.get_collection('y_conv')[0]

	correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
	true_pred = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.float32))

	for epoch in range(nb_epochs):
		print("Epoch: %d" % epoch)
		print("Learning rate: " + str(learning_rate.eval()))
		
		avg_ttl = []
		nb_true_pred = 0

		# shuffle data
		perm = np.random.permutation(n_train_samples)
		print('x_train = '+str(perm))
		if epoch % 10 == 0:
			saver.save(sess, "save/current/model.ckpt")

		for i in range(0, n_train_samples, batch_size):
		
			x_batch = train_sets.data[perm[i: (i + batch_size)]]
			#print('x_batch['+str(i)+'] = '+str(perm[i:(i+batch_size)]))

			batch_target = np.asarray(train_sets.target[perm[i:i + batch_size]])
			y_batch = np.zeros((len(x_batch), 46), dtype=np.float32)
			y_batch[np.arange(len(x_batch)), batch_target] = 1.0
			#print('batch_target = '+str(batch_target))

			ttl, _ = sess.run([cross_entropy_loss, train_step],
								  feed_dict={x: x_batch, y_: y_batch, keep_prob_fc1: (1 - drop_out_prob),
											 keep_prob_fc2: (1 - drop_out_prob)})
			
			avg_ttl.append(ttl*len(x_batch))

			nb_true_pred += true_pred.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob_fc1: 1, keep_prob_fc2: 1})
			print('Batch '+str(i)+' : Number of true prediction: '+str(nb_true_pred))
		print("Average total loss: " + str(np.sum(avg_ttl)/n_train_samples))
		print("Train accuracy: " + str(nb_true_pred * 1.0 / n_train_samples))
train()