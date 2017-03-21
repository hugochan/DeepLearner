import os
import timeit
import argparse
import pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


label_lookup = dict(zip(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], range(10)))


def revdict(d):
    """
    Reverse a dictionary mapping.
    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary).
    """
    return dict((v, k) for (k, v) in d.iteritems())

def get_all_files(in_dir, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(in_dir) for file in filenames if not file.startswith('.')]
    else:
        return [os.path.join(in_dir, filename) for filename in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, filename)) and not filename.startswith('.')]

def load_image(in_dir):
    data = []
    files = get_all_files(in_dir)
    for each in files:
        image = mpimg.imread(each).reshape((-1)) # use Pillow 2.1.0
        data.append(image)

    return np.r_[data].astype('float32')

def load_label(in_file):
    with open(in_file, 'r') as f:
        labels = f.read().strip().split('\n')

    return labels

def load_weights(file):
    try:
        with open(file, 'r') as f:
            weights = pickle.load(f)
    except Exception as e:
        raise e

    return weights

def save_weights(weights, file):
    try:
        with open(file, 'w') as f:
            pickle.dump(weights, f)
    except Exception as e:
        raise e

def one_hot_encode(labels, label_lookup):
    label_set = set(labels)
    n_labels = len(label_set)
    # label_lookup = dict(list(label_set), range(n_labels))
    label_codes = np.eye(n_labels)
    encoded_labels = []
    for each in labels:
        encoded_labels.append(label_codes[label_lookup[each]])

    return np.r_[encoded_labels]

def shuffle_data(X_data, Y_data):
    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data = X_data[idx]
    Y_data = Y_data[idx]

    return X_data, Y_data

def calc_loss(y, y_hat):
    return .5 * tf.reduce_sum(tf.squared_difference(y_hat, y)) / tf.cast(tf.shape(y)[0], tf.float32)

def feedforward(X, W1, W10, W2, W20, W3, W30):
    # h1 = relu(tf.matmul(W1, X, transpose_a=True, transpose_b=True) + W10)
    # h2 = relu(tf.matmul(W2, h1, transpose_a=True) + W20)
    # y_hat = softmax(tf.matmul(W3, h2, transpose_a=True) + W30, 0)

    # return tf.transpose(h1), tf.transpose(h2), tf.transpose(y_hat)

    # Hidden layer with RELU activation
    h1 = tf.nn.relu(tf.add(tf.matmul(X, W1), tf.transpose(W10)))
    # Hidden layer with RELU activation
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), tf.transpose(W20)))
    # Output layer with soft activation
    y_hat = tf.nn.softmax(tf.add(tf.matmul(h2, W3), tf.transpose(W30)))
    return h1, h2, y_hat

def next_batch(X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data
        yield (X[i: i + batch_size], Y[i: i + batch_size])


def run_dnn(train_data, h_dims, lr=1e-3, batch_size=50, max_epoch=1000, min_delta=1e-3, patience=10, shuffle=True, print_per_epoch=10):
    """Stochastic Gradient descent optimizer.
    """
    if len(h_dims) != 2:
        raise Exception("2 hidden layers are expected.")
    print 'train on %s samples' % (train_data.num_examples)
    n_input = 784
    n_class = 10
    X = tf.placeholder(tf.float32, shape=[None, n_input], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, n_class], name='Y')
    std = 0.1
    seed = 1234
    W1 = tf.Variable(tf.random_normal([n_input, h_dims[0]], mean=0, stddev=std, seed=seed))
    W2 = tf.Variable(tf.random_normal([h_dims[0], h_dims[1]], mean=0, stddev=std, seed=seed))
    W3 = tf.Variable(tf.random_normal([h_dims[1], n_class], mean=0, stddev=std, seed=seed))
    W10 = tf.Variable(tf.zeros([h_dims[0], 1]))
    W20 = tf.Variable(tf.zeros([h_dims[1], 1]))
    W30 = tf.Variable(tf.zeros([n_class, 1]))

    train_loss_history = []
    val_loss_history = []
    n_incr_error = 0  # nb. of consecutive increase in error
    best_loss = np.Inf
    n_batches = int(train_data.num_examples / batch_size)

    _, _, y_hat = feedforward(X, W1, W10, W2, W20, W3, W30)
    cost = calc_loss(Y, y_hat)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_epoch in range(1, max_epoch + 1):
            n_incr_error += 1
            train_loss = 0.
            val_loss = 0.

            for i in range(n_batches):
                batch_X, batch_Y = train_data.next_batch(batch_size)
                _, train_batch_loss = sess.run([optimizer, cost], {X: batch_X, Y: batch_Y})
                train_loss_history.append(train_batch_loss)
                train_loss += train_batch_loss / n_batches

            current_loss = train_loss
            if current_loss - min_delta < best_loss:
                # update best error (NLL), iteration
                best_loss = current_loss
                best_W = sess.run([W1, W10, W2, W20, W3, W30])
                best_epoch = n_epoch
                n_incr_error = 0

            if n_epoch % print_per_epoch == 0:
                print 'Epoch %s/%s, train loss: %s, val loss: %s' % (n_epoch, max_epoch, train_loss, val_loss)

            if n_incr_error >= patience:
                print 'Early stopping occured.'
                return best_W, best_loss, train_loss_history, val_loss_history

    return best_W, best_loss, train_loss_history, val_loss_history


def predict(X_data, W_val):
    X = tf.placeholder(tf.float32, shape=[None, X_data.shape[1]], name='X')
    W = [tf.Variable(each.astype('float32')) for each in W_val]
    _, _, pred = feedforward(X, W[0], W[1], W[2], W[3], W[4], W[5])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred_val = sess.run(pred, {X: X_data})

    return np.argmax(pred_val, axis=1)

def calc_clf_error(X_data, labels, W, label_lookup):
    label_lookup_rev = revdict(label_lookup)
    pred = predict(X_data, W)
    error_per_class = defaultdict(float)
    count_per_class = defaultdict(float)
    y = np.argmax(labels, 1)

    for i in range(len(y)):
        count_per_class[y[i]] += 1
        if not y[i] == pred[i]:
            error_per_class[y[i]] += 1

    for each in error_per_class:
        error_per_class[each] /= count_per_class[each]

    return dict(error_per_class), np.mean(error_per_class.values())


# def plot_weights(W):
#     n_pixel, k = W.shape
#     dim = int(np.sqrt(n_pixel - 1))
#     for i in range(k):
#         img = W[:-1, i].reshape(dim, dim)
#         plt.imshow(img)
#         plt.colorbar()
#         plt.show()

def plot_loss(train_loss, val_loss, per=5, save_file='loss.png'):
    assert len(train_loss) == len(val_loss)
    plt.figure(figsize=(10, 10), facecolor='white')
    idx = np.arange(0, len(train_loss), per)
    plt.plot(idx, train_loss[idx], linestyle='None', alpha=1.0, marker='bs', markersize=6, label='train loss')
    plt.plot(idx, val_loss[idx], linestyle='None', alpha=1.0, marker='g^', markersize=6, label='val loss')
    plt.xlabel('# of iter')
    plt.ylabel('loss')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    # plt.show()

def train(args):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    with tf.Session() as sess:
        start = timeit.default_timer()
        W, best_loss, train_loss_history, val_loss_history = run_dnn(mnist.train, \
                    args.hidden_dims, lr=args.learning_rate, batch_size=args.batch_size, \
                    max_epoch=args.max_epoch, patience=args.patience, print_per_epoch=args.print_per_epoch)

        print 'best loss: %s' % best_loss

        save_weights(W, args.save_weights)
        print 'saved model weights file to %s' % args.save_weights

        print 'runtime: %ss' % (timeit.default_timer() - start)

        # if args.plot_weights:
        #     plot_weights(W)

        if args.plot_loss:
            plot_loss(train_loss_history, val_loss_history)

def test(args):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    W = load_weights(args.load_weights)

    error_per_class, avg_error = calc_clf_error(mnist.test.images, mnist.test.labels, W, label_lookup)
    print 'error per class: %s' % error_per_class
    print 'average error: %s' % avg_error

#     # if args.plot_weights:
#     #     plot_weights(W)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('-in', '--input', required=True, type=str, help='path to the input data')
    parser.add_argument('-label', '--label', required=True, type=str, help='path to the label')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations (default 100)')
    parser.add_argument('-h_dims', '--hidden_dims', type=int, nargs='*', help='dimensions of hidden nodes')
    parser.add_argument('-n_val', '--num_validation', type=int, default=100, help='validation set size (default 100)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-bs', '--batch_size', type=int, default=50, help='batch size (default 50)')
    parser.add_argument('-p', '--patience', type=int, default=10, help='early stopping patience (default 10)')
    parser.add_argument('-pp_iter', '--print_per_epoch', type=int, default=1, help='print per iteration (default 10)')
    parser.add_argument('-sw', '--save_weights', type=str, default='weights.p', help='path to the output weights (default weights.p)')
    parser.add_argument('-lw', '--load_weights', type=str, help='path to the loaded weights')
    parser.add_argument('-pw', '--plot_weights', action='store_true', help='flag: plot weights')
    parser.add_argument('-pl', '--plot_loss', action='store_true', help='flag: plot loss')
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        if not args.load_weights:
            raise Exception('load_weights arg needed in test phase')
        test(args)

if __name__ == '__main__':
    main()
