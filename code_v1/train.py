import smile as model
import numpy as np
import os
import tensorflow as tf 
#import read_data
from read_dataset import PRMUDataSet
import pickle

nb_epochs = 100
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
		x, y_label = model.input()
		y_inference = model.inference(x)
		loss = model.loss(y_inference, y_label)
		train_step = model.train_op(loss, global_step)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
	else:
		print('Load exist model')
		saver = tf.train.import_meta_graph('save/current/model.ckpt.meta')
		saver.restore(sess, 'save/current/model.ckpt')

	#origin_loss = tf.get_collection('origin_loss')[0]
	#regul_loss = tf.get_collection('regul_loss')[0]
	total_loss = tf.get_collection('total_loss')[0]

	learning_rate = tf.get_collection('learning_rate')[0]
	train_step = tf.get_collection('train_step')[0]

	keep_prob_fc1 = tf.get_collection('keep_prob_fc1')[0]
	keep_prob_fc2 = tf.get_collection('keep_prob_fc2')[0]
	x = tf.get_collection('x')[0]
	y_label = tf.get_collection('y_')[0]
	y_inference = tf.get_collection('y_conv')[0]

	correct_prediction = tf.equal(tf.arg_max(y_inference,1),tf.arg_max(y_label,1))
	true_pred = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.float32))	

	#images_inputX, images_inputY = read_data.read_data()
	
	#np.save('image_inputX.npy',images_inputX)
	#np.save('image_inputY.npy',images_inputY)																																																																																																																																																																																																																																																																																																										
	##print('Load data')
	#images_inputX = np.load('image_inputX.npy')
	#images_inputY = np.load('image_inputY.npy')
	#train_images, train_labels, valid_images, valid_labels = images_inputX[:3000], images_inputY[:3000], images_inputX[3000:], images_inputY[3000:]

	for epoch in range(nb_epochs):
		print("Epoch: %d"%epoch)
		print("Learning rate: "+str(learning_rate.eval()))
		avg_ttl = []
		nb_true_pred = 0

		#shuffle data
		perm = np.random.permutation(n_train_samples)
		#print("img_train "+str(p))
		#avg_rgl = []
		if epoch % 10 == 0 :
			saver.save(sess,"save/current/model.ckpt")


		for i in range(0,n_train_samples,batch_size):
		#for i in range(nb_images_train//batch_size + 1) :
			'''first_img = i*batch_size
			end_img = min((i+1)*batch_size, nb_images_train)
			x_batch , y_batch = read_data.read_data(perm[first_img:end_img])
			'''
			#i=0
			#print("x_batch_id = "+str(perm[i:i+batch_size]))
			x_batch = train_sets.data[perm[i : i+batch_size]]
			#print("x_batch: "+str(x_batch))
			#x_batch = train_sets.data[0]
			#print('x_batch '+str(train_sets.target[0]))
			#print("type of x_batch "+str(type(x_batch)))
			#print("shape x_batch = "+str(x_batch.shape)+"   len x_batch = "+str(len(x_batch)))
			
			batch_target = np.asarray(train_sets.target[perm[i:i + batch_size]])
			#print("batch_target = "+str(batch_target))
			y_batch = np.zeros((len(x_batch), 46), dtype=np.float32)
			y_batch[np.arange(len(x_batch)), batch_target] = 1.0
			#print("y_batch: "+str(y_batch))
			#print("shape y_batch = "+str(y_batch.shape))

			#print('y_batch'+str(y_batch))

			ttl, _ = sess.run([total_loss,train_step],

									feed_dict = {x:x_batch, y_label:y_batch, keep_prob_fc1: (1 - drop_out_prob), keep_prob_fc2: (1 - drop_out_prob)})
			#print("len x = "+str(len(x_batch)))
			#print("batch "+str(i)+" cross_entropy = "+str(ttl))
			#avg_ttl += (ttl*len(x_batch))
			avg_ttl.append(ttl)

			nb_true_pred += true_pred.eval(feed_dict = {x:x_batch, y_label:y_batch, keep_prob_fc1: 1, keep_prob_fc2: 1})
		#avg_rgl = np.average(avg_rgl)
		#avg_ttl = avg_ttl/nb_images_train
		#avg_ttl = avg_ttl/n_train_samples

		print("Average total loss: "+str(np.sum(avg_ttl)))
		print("Train accuracy: "+str(nb_true_pred*1.0/n_train_samples))
		##print(avg_rgl)

train()