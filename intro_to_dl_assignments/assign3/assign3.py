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

def relu(X):
    return tf.nn.relu(X)

def softmax(X):
    f = tf.exp(X)

    return tf.div(f, tf.reduce_sum(f, 1, keep_dims=True))

def calc_loss(y, y_hat):
    y_diff = y_hat - y
    return .5 * tf.reduce_sum(tf.multiply(y_diff, y_diff)) / tf.cast(tf.shape(y)[0], tf.float32)

def feedforward(X, W1, W10, W2, W20, W3, W30):
    h1 = relu(tf.matmul(W1, X, transpose_a=True, transpose_b=True) + W10)
    h2 = relu(tf.matmul(W2, h1, transpose_a=True) + W20)
    y_hat = softmax(tf.matmul(W3, h2, transpose_a=True) + W30)

    return tf.transpose(h1), tf.transpose(h2), tf.transpose(y_hat)

def backprop(X, y, h1, h2, y_hat, W1, W10, W2, W20, W3, W30, lr):
    delta_y_hat = y_hat - y # bs * n_output
    n_output = int(y.get_shape()[1])
    n_h2 = int(h2.get_shape()[1])
    n_h1 = int(h1.get_shape()[1])
    n_input = int(X.get_shape()[1])
    # bs = tf.shape(X)[0]
    tmp = tf.reshape(y_hat, (-1, n_output, 1)) + tf.constant(0., shape=(1, 1, n_output)) # bs * n_output * n_output
    tmp2 = tf.eye(n_output) - tmp # bs * n_output * n_output
    tmp3 = tf.multiply(tmp, tmp2) # bs * n_output * n_output
    delta_y_hat_W3 = tf.multiply(tf.reshape(tmp3, (-1, n_output, 1, n_output)), tf.reshape(h2, (-1, 1, n_h2, 1))) # bs * n_output * n_h2 * n_output
    delta_W3 = tf.zeros((1, n_h2, n_output))
    for i in range(n_output):
        delta_W3 += delta_y_hat_W3[:, i] * tf.reshape(delta_y_hat[:, i], (-1, 1, 1))
    delta_W30 = tf.matmul(tmp3, tf.reshape(delta_y_hat, (-1, n_output, 1))) # bs * n_output * 1
    # bs * n_h2 * n_output
    delta_y_hat_h2 = tf.multiply(W3 - tf.reshape(tf.matmul(y_hat, W3, transpose_b=True), (-1, n_h2, 1)), tf.reshape(y_hat, (-1, 1, n_output)))
    delta_h2 = tf.matmul(delta_y_hat_h2, tf.reshape(delta_y_hat, (-1, n_output, 1))) # bs * n_h2 * 1

    delta_W2 = tf.zeros((1, n_h1, n_h2))
    delta_W20 = tf.zeros((1, n_h2, 1))
    h2_mask = tf.reshape(tf.to_float(h2 > 0), (-1, 1, n_h2))
    eye = tf.eye(n_h2)
    delta_h2_ext = tf.reshape(delta_h2, (-1, n_h2, 1, 1))
    for i in range(n_h2):
        mask = tf.multiply(eye[i], h2_mask)
        delta_W2 += tf.multiply(tf.multiply(tf.reshape(h1, (-1, n_h1, 1)), mask), delta_h2_ext[:, i])
        delta_W20 += tf.multiply(tf.reshape(mask, (-1, n_h2, 1)), delta_h2_ext[:, i])

    delta_h1 = tf.matmul(tf.multiply(W2, h2_mask), delta_h2) # bs * n_h1 * 1

    delta_W1 = tf.zeros((1, n_input, n_h1))
    delta_W10 = tf.zeros((1, n_h1, 1))
    h1_mask = tf.reshape(tf.to_float(h1 > 0), (-1, 1, n_h1))
    eye = tf.eye(n_h1)
    delta_h1_ext = tf.reshape(delta_h1, (-1, n_h1, 1, 1))
    for i in range(n_h1):
        mask = tf.multiply(eye[i], h1_mask)
        delta_W1 += tf.multiply(tf.multiply(tf.reshape(X, (-1, n_input, 1)), mask), delta_h1_ext[:, i])
        delta_W10 += tf.multiply(tf.reshape(mask, (-1, n_h1, 1)), delta_h1_ext[:, i])

    delta_W1 = tf.reduce_mean(delta_W1, 0)
    delta_W10 = tf.reduce_mean(delta_W10, 0)
    delta_W2 = tf.reduce_mean(delta_W2, 0)
    delta_W20 = tf.reduce_mean(delta_W20, 0)
    delta_W3 = tf.reduce_mean(delta_W3, 0)
    delta_W30 = tf.reduce_mean(delta_W30, 0)

    W1_update = W1.assign_sub(lr * delta_W1)
    W10_update = W10.assign_sub(lr * delta_W10)
    W2_update = W2.assign_sub(lr * delta_W2)
    W20_update = W20.assign_sub(lr * delta_W20)
    W3_update = W3.assign_sub(lr * delta_W3)
    W30_update = W30.assign_sub(lr * delta_W30)

    return W1_update, W10_update, W2_update, W20_update, W3_update, W30_update

def next_batch(X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data
        yield (X[i: i + batch_size], Y[i: i + batch_size])


def run_dnn(train_X, train_Y, val_X, val_Y, h_dims, lr=1e-3, batch_size=50, max_epoch=1000, patience=10, shuffle=True, print_per_epoch=10):
    """Stochastic Gradient descent optimizer.
    """
    if len(h_dims) != 2:
        raise Exception("2 hidden layers are expected.")
    print 'train on %s samples, validate on %s samples' % (train_X.shape[0], val_X.shape[0])

    X = tf.placeholder(tf.float32, shape=[None, train_X.shape[1]], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, train_Y.shape[1]], name='Y')
    W1_val = np.random.randn(train_X.shape[1], h_dims[0]).astype('float32')
    W10_val = np.zeros((h_dims[0], 1)).astype('float32')
    W2_val = np.random.randn(h_dims[0], h_dims[1]).astype('float32')
    W20_val = np.zeros((h_dims[1], 1)).astype('float32')
    W3_val = np.random.randn(h_dims[1], train_Y.shape[1]).astype('float32')
    W30_val = np.zeros((train_Y.shape[1], 1)).astype('float32')
    W1 = tf.Variable(W1_val)
    W10 = tf.Variable(W10_val)
    W2 = tf.Variable(W2_val)
    W20 = tf.Variable(W20_val)
    W3 = tf.Variable(W3_val)
    W30 = tf.Variable(W30_val)

    if shuffle:
        train_X, train_Y = shuffle_data(train_X, train_Y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_it = 0
        # train_loss_history = []
        val_loss_history = []
        n_incr_error = 0  # nb. of consecutive increase in error
        _, _, y_hat = feedforward(X, W1, W10, W2, W20, W3, W30)
        val_loss = sess.run(calc_loss(Y, y_hat), {X: val_X, Y: val_Y}) # calc init mse
        best_loss = val_loss
        best_W = [W1_val, W10_val, W2_val, W20_val, W3_val, W30_val]
        val_loss_history.append(val_loss)
        n_iter = 1
        n_batches = train_X.shape[0] / batch_size + (train_X.shape[0] % batch_size != 0)
        while True:
            n_incr_error += 1
            val_loss = 0.
            for (batch_X, batch_Y) in next_batch(train_X, train_Y, batch_size):
                # do gradient descent for one iteration
                # calc val loss
                h1, h2, y_hat = feedforward(X, W1, W10, W2, W20, W3, W30)
                weights_update = backprop(X, Y, h1, h2, y_hat, W1, W10, W2, W20, W3, W30, lr)
                W_val = sess.run(weights_update, {X: train_X, Y: train_Y})
                _, _, y_hat = feedforward(X, W1, W10, W2, W20, W3, W30)
                val_loss += sess.run(calc_loss(Y, y_hat), {X: val_X, Y: val_Y})

            val_loss /= n_batches

            val_loss_history.append(val_loss)

            if val_loss < best_loss:
                # update best error (NLL), iteration
                best_loss = val_loss
                best_W = W_val
                best_it = n_iter
                n_incr_error = 0

            if n_iter % print_per_epoch == 0:
                print 'Iter %s/%s, val loss: %s' % (n_iter, max_epoch, val_loss)

            if n_incr_error >= patience:
                print 'Early stopping occured.'
                return best_W, best_loss, val_loss_history

            if n_iter >= max_epoch:
                print 'Warning: not converged.'
                return best_W, best_loss, val_loss_history

            if shuffle:
                train_X, train_Y = shuffle_data(train_X, train_Y)
            n_iter += 1

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


# def plot_weights(W):
#     n_pixel, k = W.shape
#     dim = int(np.sqrt(n_pixel - 1))
#     for i in range(k):
#         img = W[:-1, i].reshape(dim, dim)
#         plt.imshow(img)
#         plt.colorbar()
#         plt.show()

def plot_loss(loss, per=5):
    idx = np.arange(0, len(loss), per)
    plt.plot(idx, loss[idx])
    plt.xlabel('# of iter')
    plt.ylabel('loss')
    plt.show()

def train(args):
    X = load_image(args.input) / 255.
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
        W, best_loss, val_loss_history = run_dnn(train_X, train_Y, val_X, val_Y, \
                    args.hidden_dims, lr=args.learning_rate, batch_size=args.batch_size, \
                    max_epoch=args.max_epoch, patience=10, shuffle=True, print_per_epoch=10)

        print 'best loss: %s' % best_loss

        save_weights(W, args.save_weights)
        print 'saved model weights file to %s' % args.save_weights

        print 'runtime: %ss' % (timeit.default_timer() - start)

        # if args.plot_weights:
        #     plot_weights(W)

        if args.plot_loss:
            plot_loss(val_loss_history)

# def test(args):
#     X = load_image(args.input) / 255.
#     X = np.append(X, np.ones((X.shape[0], 1)), axis=1).astype('float32')
#     labels = load_label(args.label)
#     W = load_weights(args.load_weights)

#     error_per_class, avg_error = calc_clf_error(X, labels, W, label_lookup)
#     print 'error per class: %s' % error_per_class
#     print 'average error: %s' % avg_error

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

