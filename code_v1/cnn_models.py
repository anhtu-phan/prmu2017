import tensorflow as tf
import numpy as np
import sys
from random import shuffle
from optparse import OptionParser
# Import data
from read_dataset import PRMUDataSet
import os

train_sets = PRMUDataSet("1_samp_0.9")
train_sets.load_data_target()
n_train_samples = train_sets.get_n_types_target()
print (n_train_samples)

valid_sets = PRMUDataSet("1_test_0.1")
valid_sets.load_data_target()
n_valid_samples = valid_sets.get_n_types_target()
print (n_valid_samples)

#command line options
parser = OptionParser()

#parse command line options
(options, args) = parser.parse_args()
if (len(args) != 2 or (args[0] != "test" and args[0] != "train")):
    print ("usage: -options action(test or train) gpu_id")
    print (options)
    sys.exit(2)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
learning_rate = 0.01
batch_size = 256
n_epoch = 200

# Network Parameters
n_input = 9216  # Nakayosi data input (img shape: 96*96)
n_classes = 3345  # Nakayosi total classes
dropout = 0.5  # Dropout, probability to keep units

# tf Graph input

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
phase_train = tf.placeholder(tf.bool, name='phase_train')

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def batch_norm(x, n_out, phase_train, relu = True):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        if relu == True:
            normed = tf.nn.relu(normed)
    return normed


def new_conv2d(input,  # The previous layer.
               num_input_channels,  # Num. channels in prev. layer.
               filter_size,  # Width and height of each filter.
               num_filters,
               relu=True):  # Number of filters.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    if relu == True:
        layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 dropout=0,  # Use dropout?
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    if dropout != 0:
        layer = tf.nn.dropout(layer, dropout)

    return layer


# Create model
def conv_net(x, dropout, phase_train):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 96, 96, 1])

    # Convolution Layer 1
    conv1, weights_conv1_1 = new_conv2d(input=x, num_input_channels=1, filter_size=3, num_filters=32, relu = False)
    conv1 = batch_norm(conv1, 32, phase_train, relu = True)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer 2
    conv2, weights_conv2_1 = new_conv2d(input=conv1, num_input_channels=32, filter_size=3, num_filters=32, relu = False)
    conv2 = batch_norm(conv2, 32, phase_train, relu = True)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer 3
    conv3, weights_conv3_1 = new_conv2d(input=conv2, num_input_channels=32, filter_size=3, num_filters=64, relu = False)
    conv3 = batch_norm(conv3, 64, phase_train, relu = True)
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Convolution Layer 4
    conv4, weights_conv4_1 = new_conv2d(input=conv3, num_input_channels=64, filter_size=3, num_filters=64, relu = False)
    conv4 = batch_norm(conv4, 64, phase_train, relu = True)
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)

    # Convolution Layer 5
    conv5, weights_conv5_1 = new_conv2d(input=conv4, num_input_channels=64, filter_size=3, num_filters=128, relu = False)
    conv5 = batch_norm(conv5, 128, phase_train, relu = True)
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=2)

    layer_flat, num_features = flatten_layer(conv5)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features,
                             num_outputs=4096, dropout=dropout, use_relu=True)

    layer_fc1 = new_fc_layer(input=layer_fc1, num_inputs=4096,
                             num_outputs=4096, dropout=dropout, use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=4096,
                             num_outputs=n_classes,
                             dropout=0,
                             use_relu=False)
    return layer_fc2


# Construct model
pred = conv_net(x, keep_prob, phase_train)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    if args[0] == "train":
        epoch = 1
        best_validation_accuracy = 0
        while epoch <= n_epoch:
            print("Epoch: ", str(epoch))

            perm = np.random.permutation(n_train_samples)
            for i in range(0, n_train_samples, batch_size):
                batch_x = np.asarray(train_sets.data[perm[i:i + batch_size]])
                batch_target = np.asarray(train_sets.target[perm[i:i + batch_size]])
                batch_y = np.zeros((len(batch_x), n_classes), dtype=np.float32)
                batch_y[np.arange(len(batch_x)), batch_target] = 1

                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                                   keep_prob: dropout, phase_train: True})

            epoch += 1

            # valid

            sum_valid_accuracy = 0
            sum_valid_loss = 0
            for i in range(0, n_valid_samples, batch_size):
                batch_x = np.asarray(valid_sets.data[i:i + batch_size])
                batch_target = np.asarray(valid_sets.target[i:i + batch_size])
                batch_y = np.zeros((len(batch_x), n_classes), dtype=np.float32)
                batch_y[np.arange(len(batch_x)), batch_target] = 1

                real_batchsize = len(batch_x)

                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase_train: False})
                sum_valid_loss += float(loss) * real_batchsize
                sum_valid_accuracy += float(acc) * real_batchsize

            if sum_valid_accuracy / n_valid_samples > best_validation_accuracy:
                best_validation_accuracy = sum_valid_accuracy / n_valid_samples
                improved_str = "*"
                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=sess, save_path=save_path)
            else:
                improved_str = ''
            print ("validation samples = ", n_valid_samples)
            print ("validation mean loss=", sum_valid_loss / n_valid_samples, "accuracy=", sum_valid_accuracy / n_valid_samples, improved_str)

        print ("Optimization Finished!")

    else:
        print ("Testing...!")

        # evaluation
        sum_test_accuracy = 0
        sum_test_loss = 0
        n_test_total_samples = 0
        for data_i in range(1, 6):
            test_sets = PRMUDataSet("1_test_5")
            test_sets.load_data_target()
            n_test_samples = test_sets.get_n_types_target()
            n_test_total_samples = n_test_total_samples + n_test_samples
            saver.restore(sess=sess, save_path=save_path)

            for i in range(0, n_test_samples, batch_size):
                batch_x = np.asarray(test_sets.data[i:i + batch_size])
                batch_target = np.asarray(test_sets.target[i:i + batch_size])
                batch_y = np.zeros((len(batch_x), n_classes), dtype=np.float32)
                batch_y[np.arange(len(batch_x)), batch_target] = 1

                real_batchsize = len(batch_x)

                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase_train: False})
                sum_test_loss += float(loss) * real_batchsize
                sum_test_accuracy += float(acc) * real_batchsize

        print ("Test mean loss=", sum_test_loss / n_test_total_samples, "accuracy=", sum_test_accuracy / n_test_total_samples)
        print ("Test Finished!")
