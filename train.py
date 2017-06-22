import model
import numpy as np
import os
import tensorflow as tf 
import read_data
import pickle

nb_epochs = 10
batch_size = 32 
drop_out_prob = 0.5
nb_images_train = 3000

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train():
	sess = tf.InteractiveSession()

	if not os.path.isfile('save/current/model.ckpt.index'):		
		print('Create new model')
		x, y_label = model.input()
		y_inference = model.inference(x)
		loss = model.loss(y_inference, y_label)
		train_step = model.train_op(loss)
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
	x = tf.get_collection('x')[0]
	y_label = tf.get_collection('y_label')[0]
	#images_inputX, images_inputY = read_data.read_data()
	
	#np.save('image_inputX.npy',images_inputX)
	#np.save('image_inputY.npy',images_inputY)																																																																																																																																																																																																																																																																																																										
	##print('Load data')
	#images_inputX = np.load('image_inputX.npy')
	#images_inputY = np.load('image_inputY.npy')
	#train_images, train_labels, valid_images, valid_labels = images_inputX[:3000], images_inputY[:3000], images_inputX[3000:], images_inputY[3000:]

	for epoch in range(nb_epochs):
		print("Epoch: %d"%epoch)
		print("Learning rate: %f"%learning_rate)
		avg_ttl = 0
		#shuffle data
		p = np.random.permutation(nb_images_train)
		#print("img_train "+str(p))
		#avg_rgl = []
		if epoch % 10 == 0 :
			saver.save(sess,"save/current/model.ckpt")

		for i in range(nb_images_train//batch_size + 1):
			first_image = i*batch_size
			end_image = min((i+1)*batch_size, nb_images_train)
			
			x_batch , y_batch = read_data.read_data(p[first_image:end_image])
			#print('y_batch'+str(y_batch))

			ttl, _ = sess.run([total_loss,train_step],

									feed_dict = {x:x_batch, y_label:y_batch, keep_prob_fc1: (1 - drop_out_prob)})
			avg_ttl += ttl*(end_image -  first_image)

		#avg_rgl = np.average(avg_rgl)
		avg_ttl = avg_ttl/nb_images_train

		print("Average total loss: "+str(avg_ttl))
		##print(avg_rgl)

train()