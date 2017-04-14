import os
import argparse
import timeit
import cPickle as pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn


# Network Parameters
vocab_size = 8745
emb_size = 300 # word embedding size
n_steps = 25 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 1

checkpoint_dir = 'ckpt/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Import data
def load_data(file):
    npzfile = np.load(file)
    train_x, train_y, train_mask = npzfile["train_x"], npzfile["train_y"], npzfile["train_mask"]
    val_x, val_y, val_mask = npzfile["val_x"], npzfile["val_y"], npzfile["val_mask"]
    return (train_x, train_y, train_mask), (val_x, val_y, val_mask)

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

class RNN(object):
    def __init__(self):

        # tf Graph input
        self.x = tf.placeholder("int32", [None, n_steps])
        self.mask = tf.placeholder("float", [None, n_steps])
        self.y = tf.placeholder("float", [None, n_classes])

        # Define weights
        self.weights = {
            'emb': tf.get_variable('emb', shape=(vocab_size, emb_size), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('out', shape=(n_hidden, n_classes), initializer=tf.contrib.layers.xavier_initializer())
        }
        self.biases = {
            'out': tf.Variable(tf.constant(0., shape=[n_classes]))
        }

        # Construct model
        self.y_hat = self.run_rnn(self.x, self.weights, self.biases)
        self.pred = tf.where(tf.greater_equal(self.y_hat, tf.zeros_like(self.y_hat)), tf.ones_like(self.y_hat), tf.zeros_like(self.y_hat))

    def run_rnn(self, x, weights, biases):
        rnn_input = tf.nn.embedding_lookup(weights['emb'], x)
        lstm_cell = rnn.LSTMCell(n_hidden)

        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, rnn_input, dtype=tf.float32)
        last_relevant_state = states[1]
        # Linear activation, using rnn inner loop last output
        return tf.matmul(last_relevant_state, weights['out']) + biases['out']


    def train(self, train_x, train_y, mask_x, val_x, val_y, mask_y, lr=1e-3, batch_size=128, max_epoch=30, min_delta=1e-4, patience=10, print_per_epoch=10, out_model='my_model'):
        print 'Train on %s samples, validate on %s samples' % (train_x.shape[0], val_x.shape[0])
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(self.pred, self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        n_incr_error = 0  # nb. of consecutive increase in error
        best_loss = np.Inf
        n_batches = train_x.shape[0] / batch_size + (train_x.shape[0] % batch_size != 0)

        # Create the collection
        tf.get_collection("validation_nodes")
        # Add stuff to the collection.
        tf.add_to_collection("validation_nodes", self.x)
        tf.add_to_collection("validation_nodes", self.mask)
        tf.add_to_collection("validation_nodes", self.pred)
        saver = tf.train.Saver(max_to_keep=1)

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

                    train_loss_history.append(train_batch_loss)
                    train_acc_history.append(train_batch_acc)
                    val_loss_history.append(val_batch_loss)
                    val_acc_history.append(val_batch_acc)

                if val_loss - min_delta < best_loss:
                    best_loss = val_loss
                    save_path = saver.save(sess, checkpoint_dir + out_model, global_step=n_epoch)
                    print "Model saved in file: %s" % save_path
                    n_incr_error = 0

                if n_epoch % print_per_epoch == 0:
                    print 'Epoch %s/%s, train loss: %.5f, train acc: %.5f, val loss: %.5f, val acc: %.5f' % \
                                                (n_epoch, max_epoch, train_loss, train_acc, val_loss, val_acc)

                if n_incr_error >= patience:
                    print 'Early stopping occured. Optimization Finished!'
                    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

            return train_loss_history, train_acc_history, val_loss_history, val_acc_history

    def pred(self, x, mask, sess):
        y = sess.run(self.pred, feed_dict={self.x: x, self.mask: mask})
        return y

    def calc_acc(self, x, y, mask, sess):
        pred = sess.run(self.pred, feed_dict={self.x: x, self.mask: mask})
        correct_pred = tf.equal(pred, y)
        accuracy = sess.run(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))
        return accuracy

    def restore_model(self, mod_file): # should no include .meta
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, mod_file)
        return sess

def plot_loss(loss, train_acc, test_acc, start=0, per=1, save_file='loss.png'):
    plt.figure(figsize=(10, 10), facecolor='white')
    idx = np.arange(start, len(loss), per)
    plt.plot(idx, loss[idx], alpha=1.0, label='loss')
    plt.plot(idx, train_acc[idx], alpha=1.0, label='train acc')
    plt.plot(idx, test_acc[idx], alpha=1.0, label='test acc')
    plt.xlabel('# of iter')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    # plt.show()

def train(args):
    (train_x, train_y, train_mask), (val_x, val_y, val_mask) = load_data(args.input)
    train_y = np.reshape(train_y, (-1, 1))
    val_y = np.reshape(val_y, (-1, 1))
    start = timeit.default_timer()
    model = RNN()
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = model.train(train_x, train_y, train_mask, val_x, val_y, val_mask,\
                                            lr=args.learning_rate, \
                                            batch_size=args.batch_size, \
                                            max_epoch=args.max_epoch, \
                                            min_delta=1e-4, \
                                            patience=args.patience, \
                                            print_per_epoch=args.print_per_epoch,
                                            out_model=args.save_model)

    print 'runtime: %.1fs' % (timeit.default_timer() - start)
    import pdb;pdb.set_trace()
    if args.plot_loss:
        plot_loss(np.array(train_loss_hist), np.array(train_acc_hist), np.array(test_acc_hist), start=0, per=10, save_file=args.plot_loss)

def test(args):
    _, (test_x, test_y, test_mask) = load_data(args.input)
    print 'Report results on %s test samples' % test_x.shape[0]
    test_y = np.reshape(test_y, (-1, 1))

    model = RNN()
    sess = model.restore_model(args.load_model)
    acc = model.calc_acc(test_x, test_y, test_mask, sess)
    print 'Accuracy: %.5f' % acc
    sess.close()

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
