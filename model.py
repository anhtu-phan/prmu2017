import tensorflow as tf
import numpy as np
from read_dataset import PRMUDataSet
import os

train_sets = PRMUDataSet("1_train_0.9")
train_sets.load_data_target()
n_train_samples = train_sets.get_n_types_target()
print ("n_train_samples = "+str(n_train_samples))


valid_sets = PRMUDataSet('1_test_0.9')
valid_sets.load_data_target()
n_valid_samples = valid_sets.get_n_types_target()
print("n_valid_samples = "+str(n_valid_samples))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

img_size = 96
started_learning_rate = 0.0005
nb_epochs = 1000
batch_size = 256 
drop_out_prob = 0.5
#lamb = 0.01

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
     
    	return layer, weight

keep_prob_fc1 = tf.placeholder(tf.float32)
keep_prob_fc2 = tf.placeholder(tf.float32)

def inference(x):
    x_image = tf.reshape(x,[-1, img_size, img_size, 1])
    #print('x_image = '+str(x_image.get_shape()))
    #l2_loss = tf.constant(0.0,dtype=tf.float32)

    conv1, W_conv1 = new_conv2d(pre_layer=x_image, filter_size=3, nb_chanels_input=1, nb_filters=32, relu=False, n_scope='conv1')
    #print('conv1 = '+str(conv1.get_shape())+"   W_conv1 = "+str(W_conv1.get_shape()))
    #l2_loss += tf.nn.l2_loss(W_conv1)

    conv1 = new_maxpool2d(conv1)
    #print('conv1_pool = '+str(conv1.get_shape()))

    conv2, W_conv2 = new_conv2d(pre_layer=conv1, filter_size=3, nb_chanels_input=32, nb_filters=32, relu=False, n_scope='conv2')
    #print('conv2 = '+str(conv2.get_shape())+"   W_conv2 = "+str(W_conv2.get_shape()))
    #l2_loss += tf.nn.l2_loss(W_conv2)

    conv2 = new_maxpool2d(conv2)
    #print('conv2_pool = '+str(conv2.get_shape()))

    conv3, W_conv3 = new_conv2d(pre_layer=conv2, filter_size=3, nb_chanels_input=32, nb_filters=64, relu=False, n_scope='conv3')
    #print('conv3 = '+str(conv3.get_shape())+"   W_conv3 = "+str(W_conv3.get_shape()))
    #l2_loss += tf.nn.l2_loss(W_conv3)

    conv3 = new_maxpool2d(conv3)
    #print('conv3_pool = '+str(conv3.get_shape()))

    conv4, W_conv4 = new_conv2d(pre_layer=conv3, filter_size=3, nb_chanels_input=64, nb_filters=64, relu=False, n_scope='conv4')
    #print('conv4 = '+str(conv4.get_shape())+"   W_conv4="+str(W_conv4.get_shape()))
    #l2_loss += tf.nn.l2_loss(W_conv4)

    conv4 = new_maxpool2d(conv4)
    #print('conv4_pool = '+str(conv4.get_shape()))
    
    conv5, W_conv5 = new_conv2d(pre_layer=conv4, filter_size=3, nb_chanels_input=64, nb_filters=128, relu=False, n_scope='conv5')
    #print('conv5 = '+str(conv5.get_shape())+"   W_conv5="+str(W_conv5.get_shape()))
    #l2_loss += tf.nn.l2_loss(W_conv5)

    conv5 = new_maxpool2d(conv5)
    #print('conv5_pool = '+str(conv5.get_shape()))

    layer_flat, nb_dim = flatten_layer(conv5)
    #print('layer_flat = '+str(layer_flat.get_shape())+'     nb_dim = '+str(nb_dim))

    fc1, W_fc1 = new_fc_layer(pre_layer=layer_flat, nb_inputs=nb_dim, nb_outputs= 4096, n_scope='fc1')
    #print('fc1 = '+str(fc1.get_shape()))
    #l2_loss += tf.nn.l2_loss(W_fc1)

    fc1_drop = tf.nn.dropout(fc1, keep_prob_fc1)
    #print('fc1_drop = '+fc1_drop.get_shape())

    fc2, W_fc2 = new_fc_layer(pre_layer=fc1_drop, nb_inputs=4096, nb_outputs=4096, n_scope='fc2')
    #print('fc2 = '+fc2.get_shape())
    #l2_loss += tf.nn.l2_loss(W_fc2)

    fc2_drop = tf.nn.dropout(fc2, keep_prob_fc2)
    #print('fc2_drop = '+fc2_drop.get_shape())
    y_conv, W_softmax = new_fc_layer(pre_layer=fc2_drop, nb_inputs=4096, nb_outputs=46, relu= False, n_scope='softmax')
    #print('y_conv = '+str(y_conv.get_shape()))

    #l2_loss = lamb*l2_loss

    return y_conv

x = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
y_ = tf.placeholder(tf.float32, [None, 46])

y_conv = inference(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#loss = cross_entropy_loss + l2_loss
#global_step = tf.contrib.framework.get_or_create_global_step()
#learning_rate = tf.train.exponential_decay(started_learning_rate, global_step, 100000, 0.96, staircase=False)

optimizer = tf.train.AdamOptimizer(learning_rate=started_learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
true_pred = tf.reduce_sum(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

'''
saver = tf.train.Saver()
save_dir = 'save/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir,'best_validation')
'''
with tf.Session() as sess:
    #best_validation_accuracy = 0
    sess.run(init)
    for epoch in range(nb_epochs) :
        print('Epoch '+str(epoch)+'/'+str(nb_epochs))
        #print('learning rate: '+str(learning_rate.eval()))
        perm = np.random.permutation(n_train_samples)
        #perm2 = np.random.permutation(n_valid_samples)
        tmp_loss = 0.0
        nb_true_pred = 0.0
        #tmp_l2_loss = []

        for i in range(0, n_train_samples, batch_size):
            x_batch = np.asarray(train_sets.data[perm[i:(i+batch_size)]])
            batch_target = np.asarray(train_sets.target[perm[i:(i+batch_size)]])
            y_batch = np.zeros((len(x_batch), 46), dtype=np.float32)
            y_batch[np.arange(len(x_batch)), batch_target] = 1.0
            #print('x_batch = '+str(x_batch.shape))
            tmp_l,tmp_op = sess.run([loss,optimizer], feed_dict ={x: x_batch, y_: y_batch, keep_prob_fc1 : 1-drop_out_prob, keep_prob_fc2 : 1- drop_out_prob})

            tmp_loss += float(tmp_l) * len(x_batch)
            #tmp_l2_loss.append(tmp_l2_ls)
            nb_true_pred += true_pred.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob_fc1 : 1, keep_prob_fc2 : 1})

        train_loss = tmp_loss/n_train_samples
        #train_l2_loss = np.average(tmp_l2_loss)
        train_acc = nb_true_pred/n_train_samples
        #print('loss: '+str(train_loss)+' - l2-loss: '+str(train_l2_loss)+' - accurancy: '+str(train_acc))
        print('loss: '+str(train_loss)+' - accurancy: '+str(train_acc))
        
        nb_valid_acc = 0.0
        nb_valid_loss = 0.0
        for i in range(0, n_valid_samples, batch_size):
            x_batch = np.asarray(valid_sets.data[perm2[i:(i+batch_size)]])
            batch_target = np.asarray(valid_sets.target[perm2[i:(i+batch_size)]])
            y_batch = np.zeros((len(x_batch),46), dtype=np.float32)
            y_batch[np.arange(len(x_batch)), batch_target] = 1.0

            valid_loss = sess.run(loss, feed_dict={x: x_batch, y_: y_batch, keep_prob_fc1 : 1, keep_prob_fc2 : 1})
            nb_valid_loss += float(valid_loss)*len(x_batch)
            nb_valid_acc += true_pred.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob_fc1 : 1, keep_prob_fc2 : 1})

        nb_valid_acc = nb_valid_acc/n_valid_samples
        nb_valid_loss = nb_valid_loss/n_valid_samples
        print('valid_loss: '+str(nb_valid_loss)+' - valid_acc: '+str(nb_valid_acc))
        
        '''if nb_valid_acc > best_validation_accuracy :
            best_validation_accuracy = nb_valid_acc
            saver.save(sess=sess, save_path=save_path)'''