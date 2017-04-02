import os
import argparse
import timeit
import cPickle as pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Network Parameters
img_size = 32
n_channel = 3
n_input = img_size * img_size * n_channel
n_classes = 10
conv1_size = 5
conv1_stride = 1
conv1_out = 32
pool1_size = 2
pool1_stride = 2
conv2_size = 5
conv2_stride = 1
conv2_out = 32
pool2_size = 2
pool2_stride = 2
conv3_size = 3
conv3_stride = 1
conv3_out = 64
fc_size = 3 * 3 * 64
checkpoint_dir = 'ckpt/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Import data
def load_data(file):
    try:
        data = pickle.load(open(file, 'r'))
    except Exception as e:
        raise e
    return data

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def shuffle_data(X_data, Y_data):
    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data = X_data[idx]
    Y_data = Y_data[idx]
    return X_data, Y_data

def next_batch(X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data
        yield (X[i: i + batch_size], Y[i: i + batch_size])

class ConvNet(object):
    def __init__(self):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])
        # Store layers weight & bias
        self.weights = {
            'wc1': tf.get_variable('wc1', shape=(conv1_size, conv1_size, n_channel, conv1_out), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('wc2', shape=(conv2_size, conv2_size, conv1_out, conv2_out), initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable('wc3', shape=(conv3_size, conv3_size, conv2_out, conv3_out), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('out', shape=(fc_size, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
        }

        self.biases = {
            'bc1': tf.Variable(tf.constant(0.1, shape=[conv1_out])),
            'bc2': tf.Variable(tf.constant(0.1, shape=[conv2_out])),
            'bc3': tf.Variable(tf.constant(0.1, shape=[conv3_out])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
        }

        # Construct model
        self.pred = self.conv_net(self.x, self.weights, self.biases)

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k, strides):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                              padding='VALID')

    # Create model
    def conv_net(self, x, weights, biases):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, img_size, img_size, n_channel])

        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'], strides=conv1_stride)
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=pool1_size, strides=pool1_stride)

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'], strides=conv2_stride)
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=pool2_size, strides=pool2_stride)

        # Convolution Layer
        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'], strides=conv3_stride)

        # Fully connected layer
        # Reshape conv3 output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, fc_size])
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def train(self, train_x, train_y, val_x, val_y, lr=1e-3, batch_size=128, max_epoch=1000, min_delta=1e-4, patience=10, print_per_epoch=10, out_model='my_model'):
        print 'Train on %s samples, validate on %s samples' % (train_x.shape[0], val_x.shape[0])
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        n_incr_error = 0  # nb. of consecutive increase in error
        best_loss = np.Inf
        n_batches = train_x.shape[0] / batch_size + (train_x.shape[0] % batch_size != 0)
        saver = tf.train.Saver(max_to_keep=patience * 2)

        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Keep training until reach max iterations
            for n_epoch in range(1, max_epoch + 1):
                n_incr_error += 1
                train_loss = 0.
                val_loss = 0.
                train_acc = 0.
                val_acc = 0.
                train_x, train_y = shuffle_data(train_x, train_y)
                for batch_x, batch_y in next_batch(train_x, train_y, batch_size):
                    # Run optimization op (backprop)
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                    train_batch_loss, train_batch_acc = sess.run([self.cost, accuracy], feed_dict={self.x: batch_x, self.y: batch_y})
                    train_loss += train_batch_loss / n_batches
                    train_acc += train_batch_acc / n_batches
                    val_batch_loss, val_batch_acc = sess.run([self.cost, accuracy], feed_dict={self.x: val_x, self.y: val_y})
                    val_loss += val_batch_loss / n_batches
                    val_acc += val_batch_acc / n_batches

                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                if val_loss - min_delta < best_loss:
                    best_loss = val_loss
                    self.save_model(saver, sess, n_epoch, checkpoint_dir + out_model)
                    n_incr_error = 0

                if n_epoch % print_per_epoch == 0:
                    print 'Epoch %s/%s, train loss: %.5f, train acc: %.5f, val loss: %.5f, val acc: %.5f' % \
                                                (n_epoch, max_epoch, train_loss, train_acc, val_loss, val_acc)

                if n_incr_error >= patience:
                    print 'Early stopping occured. Optimization Finished!'
                    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

            return train_loss_history, train_acc_history, val_loss_history, val_acc_history

    def pred(self, x, sess):
        y = sess.run(self.pred, feed_dict={self.x: x})
        return y

    def calc_clf_error(self, x, y, sess):
        pred = np.argmax(sess.run(self.pred, feed_dict={self.x: x}), 1)
        labels = np.argmax(y, 1)
        error_per_class = defaultdict(float)
        count_per_class = defaultdict(float)

        for i in range(labels.shape[0]):
            count_per_class[labels[i]] += 1
            if not labels[i] == pred[i]:
                error_per_class[labels[i]] += 1

        for each in error_per_class:
            error_per_class[each] /= count_per_class[each]

        return dict(error_per_class), np.mean(error_per_class.values())

    def calc_acc(self, x, y, sess):
        pred = sess.run(self.pred, feed_dict={self.x: x})
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = sess.run(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))
        return accuracy

    def restore_model(self, mod_file): # should no include .meta
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, mod_file)
        return sess

    def save_model(self, saver, sess, ckpt_id, out):
        # Create the collection
        tf.get_collection("validation_nodes")
        # Add stuff to the collection.
        tf.add_to_collection("validation_nodes", self.x)
        tf.add_to_collection("validation_nodes", tf.argmax(self.pred, 1))
        save_path = saver.save(sess, out, global_step=ckpt_id)
        print "Model saved in file: %s" % save_path

    def feature_visualize(self, sess, layer_ids=[0], channel_ids=[0], out='filter.png'):
        graph = tf.get_default_graph()
        graph_def = tf.GraphDef()
        layers = [op.name for op in graph.get_operations() if op.type=='Conv2D']
        # feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
        # print 'Number of layers', len(layers)
        # print 'Total number of feature channels:', sum(feature_nums)

        # start with a gray image with a little noise
        for i in layer_ids:
            for j in channel_ids:
                img_noise = np.random.uniform(size=(1, n_input))
                name = os.path.splitext(out)
                render_naive(sess, self.x, T(graph, layers[i])[:,:,:,j], img0=img_noise, \
                    iter_n=100, step=.5, out="%s_conv2d_%s_channel_%s"%(name[0], i, j)+name[1])

def plot_loss(train_loss, val_loss, start=0, per=1, save_file='loss.png'):
    assert len(train_loss) == len(val_loss)
    plt.figure(figsize=(10, 10), facecolor='white')
    idx = np.arange(start, len(train_loss), per)
    plt.plot(idx, train_loss[idx], alpha=1.0, label='train loss')
    plt.plot(idx, val_loss[idx], alpha=1.0, label='test loss')
    plt.xlabel('# of epoch')
    plt.ylabel('loss')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    # plt.show()

def train(args):
    train_x, train_y, test_x, test_y = load_data(args.input)
    train_x = np.reshape(train_x, (train_x.shape[0], -1))
    train_x = train_x.astype('float32') / 255. # scaling
    train_x = train_x - np.mean(train_x, axis=0) # normalizing
    train_y = dense_to_one_hot(np.array(train_y), n_classes)

    test_x = np.reshape(test_x, (test_x.shape[0], -1))
    test_x = test_x.astype('float32') / 255. # scaling
    test_x = test_x - np.mean(test_x, axis=0) # normalizing
    test_y = dense_to_one_hot(np.array(test_y), n_classes)

    start = timeit.default_timer()
    cnn = ConvNet()
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = cnn.train(train_x, train_y, test_x, test_y, \
                                            lr=args.learning_rate, \
                                            batch_size=args.batch_size, \
                                            max_epoch=args.max_epoch, \
                                            min_delta=1e-4, \
                                            patience=args.patience, \
                                            print_per_epoch=args.print_per_epoch,
                                            out_model=args.save_model)

    print 'runtime: %.1fs' % (timeit.default_timer() - start)

    if args.plot_loss:
        plot_loss(np.array(train_loss_hist), np.array(test_loss_hist), start=0, per=1, save_file=args.plot_loss)

def test(args):
    _, _, test_x, test_y = load_data(args.input)
    test_x = np.reshape(test_x, (test_x.shape[0], -1))
    test_x = test_x.astype('float32') / 255. # scaling
    test_x = test_x - np.mean(test_x, axis=0) # normalizing
    test_y = dense_to_one_hot(np.array(test_y), n_classes)
    print 'Report results on %s test samples' % test_x.shape[0]

    cnn = ConvNet()
    sess = cnn.restore_model(args.load_model)
    if args.plot_filter:
        cnn.feature_visualize(sess, layer_ids=[0], channel_ids=range(conv1_out), out=args.plot_filter)
    acc = cnn.calc_acc(test_x, test_y, sess)
    print 'Accuracy: %.5f' % acc
    error_per_class, avg_error = cnn.calc_clf_error(test_x, test_y, sess)
    print 'Error per class: %s' % error_per_class
    print 'Average error: %.5f' % avg_error
    sess.close()

# Let's start with a naive way of visualizing these. Image-space gradient ascent!
# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

def T(graph, layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("%s:0"%layer)

def render_naive(sess, t_input, t_obj, img0, iter_n=20, step=1.0, out='filter.png'):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # normalizing the gradient, so the same step size should work
        g /= g.std() + 1e-8         # for different layers and networks
        img += g * step
    a = np.uint8(np.clip(visstd(img.reshape((img_size, img_size, n_channel))), 0, 1) * 255)
    plt.imshow(a)
    plt.savefig(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the input data')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations (default 100)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size (default 50)')
    parser.add_argument('-p', '--patience', type=int, default=10, help='early stopping patience (default 10)')
    parser.add_argument('-pp_iter', '--print_per_epoch', type=int, default=1, help='print per iteration (default 10)')
    parser.add_argument('-sm', '--save_model', type=str, default='my_model', help='path to the output model (default my_model)')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the loaded model')
    parser.add_argument('-pf', '--plot_filter', type=str, default='filter.png', help='plot filters')
    parser.add_argument('-pl', '--plot_loss', type=str, default='loss.png', help='plot loss')
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        if not args.load_model:
            raise Exception('load_model arg needed in test phase')
        test(args)

if __name__ == '__main__':
    main()
