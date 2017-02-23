import os
import timeit
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


label_lookup = dict(zip(['1', '2', '3', '4', '5'], range(5)))


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

def softmax(X, W):
    f = tf.exp(tf.matmul(X, W))

    return tf.div(f, tf.reduce_sum(f, 1, keep_dims=True))

def calc_loss(X, Y, W, reg=1e-4):
    """Compute the Negative Log Conditional Likelihood.
    """

    return -tf.reduce_sum(tf.multiply(Y, tf.log(softmax(X, W)))) + reg * tf.reduce_sum(tf.square(W))

def eval_tensor(fetches, feed_dict):
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        return sess.run(fetches, feed_dict=feed_dict)

def grad_op(X, Y, W, reg=1e-4, lr=1e-3):
    """Do gradient descent for one iteration.
    """
    grad = -tf.matmul(X, Y - softmax(X, W), transpose_a=True) + 2 * reg * W
    W -= lr * grad

    return W

def next_batch(X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data
        yield (X[i: i + batch_size], Y[i: i + batch_size])


def sgd_optimizer(train_X, train_Y, val_X, val_Y, lr=1e-3, batch_size=50, max_iter=1000, reg=1e-4, patience=10, shuffle=True, print_per_iter=10):
    """Stochastic Gradient descent optimizer.
    """
    print 'train on %s samples, validate on %s samples' % (train_X.shape[0], val_X.shape[0])

    X = tf.placeholder(tf.float32, shape=[None, train_X.shape[1]], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, train_Y.shape[1]], name='Y')
    W = tf.random_normal([train_X.shape[1], train_Y.shape[1]], mean=0., stddev=0.01) # initialize Theta
    if shuffle:
        train_X, train_Y = shuffle_data(train_X, train_Y)

    best_it = 0
    # train_loss_history = []
    val_loss_history = []
    n_incr_error = 0  # nb. of consecutive increase in error
    val_loss, W_val = eval_tensor([calc_loss(X, Y, W, reg), W], {X: val_X, Y: val_Y}) # calc init mse
    best_loss = val_loss
    best_W = W_val
    val_loss_history.append(val_loss)

    n_iter = 1
    while True:
        for (batch_X, batch_Y) in next_batch(train_X, train_Y, batch_size):
            n_incr_error += 1
            # do gradient descent for one iteration
            W = grad_op(X, Y, tf.Variable(W_val), reg=reg, lr=lr)
            W_val = eval_tensor(W, {X: batch_X, Y: batch_Y})
            # calc val loss
            val_loss = eval_tensor(calc_loss(X, Y, tf.Variable(W_val), reg), {X: val_X, Y: val_Y})
            val_loss_history.append(val_loss)


            if val_loss < best_loss:
                # update best error (NLL), iteration
                best_loss = val_loss
                best_W = W_val
                best_it = n_iter
                n_incr_error = 0

            if n_iter % print_per_iter == 0:
                print 'Iter %s/%s, val loss: %s' % (n_iter, max_iter, val_loss)

            if n_incr_error >= patience:
                print 'Early stopping occured.'
                return best_W, best_loss, val_loss_history

            if n_iter >= max_iter:
                print 'Warning: not converged.'
                return best_W, best_loss, val_loss_history

            n_iter += 1

        if shuffle:
            train_X, train_Y = shuffle_data(train_X, train_Y)

def predict(X_data, W):
    X = tf.placeholder(tf.float32, shape=[None, X_data.shape[1]], name='X')
    W = tf.Variable(W)
    pred = softmax(X, W)
    pred_val = eval_tensor(pred, {X: X_data})

    return np.argmax(pred_val, axis=1)

def calc_clf_error(X_data, labels, W, label_lookup):
    label_lookup_rev = revdict(label_lookup)
    pred = predict(X_data, W)
    error_per_class = defaultdict(float)
    count_per_class = defaultdict(float)

    for i in range(len(labels)):
        count_per_class[labels[i]] += 1.
        if not labels[i] == label_lookup_rev[pred[i]]:
            error_per_class[labels[i]] += 1.

    for each in error_per_class:
        error_per_class[each] /= count_per_class[each]

    return dict(error_per_class), np.mean(error_per_class.values())


def plot_weights(W):
    n_pixel, k = W.shape
    dim = int(np.sqrt(n_pixel - 1))
    for i in range(k):
        img = W[:-1, i].reshape(dim, dim)
        plt.imshow(img)
        plt.colorbar()
        plt.show()

def plot_loss(loss, per=5):
    idx = np.arange(0, len(loss), per)
    plt.plot(idx, loss[idx])
    plt.xlabel('# of iter')
    plt.ylabel('loss')
    plt.show()

def train(args):
    X = load_image(args.input) / 255.
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1).astype('float32')
    labels = load_label(args.label)
    Y = one_hot_encode(labels, label_lookup)

    np.random.seed(0)
    idx = np.arange(X.shape[0])
    val_idx = np.random.choice(idx, args.num_validation, replace=False)
    train_idx = list(set(idx) - set(val_idx))
    train_X = X[train_idx]
    train_Y = Y[train_idx]
    val_X = X[val_idx]
    val_Y = Y[val_idx]


    with tf.Session() as sess:
        start = timeit.default_timer()
        W, best_loss, val_loss_history = sgd_optimizer(train_X, train_Y, val_X, val_Y,\
                lr=args.learning_rate, batch_size=args.batch_size, max_iter=args.max_iteration, \
                reg=args.regularization, patience=args.patience,\
                print_per_iter=args.print_per_iter)

        print 'best loss: %s' % best_loss

        save_weights(W, args.save_weights)
        print 'saved model weights file to %s' % args.save_weights

        print 'runtime: %ss' % (timeit.default_timer() - start)

        if args.plot_weights:
            plot_weights(W)

        if args.plot_loss:
            plot_loss(val_loss_history)

def test(args):
    X = load_image(args.input) / 255.
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1).astype('float32')
    labels = load_label(args.label)
    W = load_weights(args.load_weights)

    error_per_class, avg_error = calc_clf_error(X, labels, W, label_lookup)
    print 'error per class: %s' % error_per_class
    print 'average error: %s' % avg_error

    if args.plot_weights:
        plot_weights(W)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('-in', '--input', required=True, type=str, help='path to the input data')
    parser.add_argument('-label', '--label', required=True, type=str, help='path to the label')
    parser.add_argument('-max_iter', '--max_iteration', type=int, default=100, help='max number of iterations (default 1000)')
    parser.add_argument('-n_val', '--num_validation', type=int, default=100, help='validation set size (default 100)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=50, help='batch size (default 50)')
    parser.add_argument('-reg', '--regularization', type=float, default=1e-4, help='regularization factor (default 1e-5)')
    parser.add_argument('-p', '--patience', type=int, default=10, help='early stopping patience (default 10)')
    parser.add_argument('-pp_iter', '--print_per_iter', type=int, default=10, help='print per iteration')
    parser.add_argument('-sw', '--save_weights', type=str, default='weights.p', help='path to the output weights')
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
