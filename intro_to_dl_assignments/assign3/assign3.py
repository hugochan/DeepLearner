import os
import timeit
import argparse
import pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


n_class = 10

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

  return labels_one_hot

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
        labels = np.array([int(x) for x in f.read().strip().split('\n')])

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
            pickle.dump(weights, f, protocol=2)
    except Exception as e:
        raise e

# def one_hot_encode(labels, label_lookup):
#     label_set = set(labels)
#     n_labels = len(label_set)
#     # label_lookup = dict(list(label_set), range(n_labels))
#     label_codes = np.eye(n_labels)
#     encoded_labels = []
#     for each in labels:
#         encoded_labels.append(label_codes[label_lookup[each]])

#     return np.r_[encoded_labels]

def shuffle_data(X_data, Y_data):
    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data = X_data[idx]
    Y_data = Y_data[idx]

    return X_data, Y_data

def calc_loss(y, y_hat):
    return .5 * tf.reduce_sum(tf.squared_difference(y_hat, y)) / tf.cast(tf.shape(y)[0], tf.float32)

def feedforward(X, W1, W10, W2, W20, W3, W30):
    # Hidden layer with RELU activation
    h1 = tf.nn.relu(tf.add(tf.matmul(X, W1), tf.transpose(W10)))
    # Hidden layer with RELU activation
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), tf.transpose(W20)))
    # Output layer with soft activation
    y_hat = tf.nn.softmax(tf.add(tf.matmul(h2, W3), tf.transpose(W30)))

    return h1, h2, y_hat

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

    W1_update = W1.assign_sub(lr * tf.reduce_mean(delta_W1, 0))
    W10_update = W10.assign_sub(lr * tf.reduce_mean(delta_W10, 0))
    W2_update = W2.assign_sub(lr * tf.reduce_mean(delta_W2, 0))
    W20_update = W20.assign_sub(lr * tf.reduce_mean(delta_W20, 0))
    W3_update = W3.assign_sub(lr * tf.reduce_mean(delta_W3, 0))
    W30_update = W30.assign_sub(lr * tf.reduce_mean(delta_W30, 0))

    return W1_update, W10_update, W2_update, W20_update, W3_update, W30_update

def next_batch(X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data
        yield (X[i: i + batch_size], Y[i: i + batch_size])


def run_dnn(train_X, train_Y, val_X, val_Y, h_dims, lr=1e-3, batch_size=50, max_epoch=1000, min_delta=1e-4, patience=10, shuffle=True, print_per_epoch=10):
    """Stochastic Gradient descent optimizer.
    """
    if len(h_dims) != 2:
        raise Exception("2 hidden layers are expected.")
    print 'train on %s samples, validate on %s samples' % (train_X.shape[0], val_X.shape[0])

    X = tf.placeholder(tf.float32, shape=[None, train_X.shape[1]], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, train_Y.shape[1]], name='Y')
    std = 0.1
    seed = 1234
    W1 = tf.Variable(tf.random_normal([train_X.shape[1], h_dims[0]], mean=0, stddev=std, seed=seed))
    W2 = tf.Variable(tf.random_normal([h_dims[0], h_dims[1]], mean=0, stddev=std, seed=seed))
    W3 = tf.Variable(tf.random_normal([h_dims[1], train_Y.shape[1]], mean=0, stddev=std, seed=seed))
    W10 = tf.Variable(tf.zeros([h_dims[0], 1]))
    W20 = tf.Variable(tf.zeros([h_dims[1], 1]))
    W30 = tf.Variable(tf.zeros([train_Y.shape[1], 1]))

    train_loss_history = []
    val_loss_history = []
    n_incr_error = 0  # nb. of consecutive increase in error
    best_loss = np.Inf
    n_batches = train_X.shape[0] / batch_size + (train_X.shape[0] % batch_size != 0)

    h1, h2, y_hat = feedforward(X, W1, W10, W2, W20, W3, W30)
    cost = calc_loss(Y, y_hat)
    backprop_op = backprop(X, Y, h1, h2, y_hat, W1, W10, W2, W20, W3, W30, lr)

    if shuffle:
        train_X, train_Y = shuffle_data(train_X, train_Y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import pdb;pdb.set_trace()
        for n_epoch in range(1, max_epoch + 1):
            n_incr_error += 1
            train_loss = 0.
            val_loss = 0.
            for (batch_X, batch_Y) in next_batch(train_X, train_Y, batch_size):
                sess.run(backprop_op, {X: batch_X, Y: batch_Y})
                train_batch_loss = sess.run(cost, {X: batch_X, Y: batch_Y})
                val_batch_loss = sess.run(cost, {X: val_X, Y: val_Y})
                train_loss_history.append(train_batch_loss)
                val_loss_history.append(val_batch_loss)
                train_loss += train_batch_loss / n_batches
                val_loss += val_batch_loss / n_batches

            current_loss = val_loss
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

            if shuffle:
                train_X, train_Y = shuffle_data(train_X, train_Y)
    return best_W, best_loss, train_loss_history, val_loss_history

def predict(X_data, W_val):
    X = tf.placeholder(tf.float32, shape=[None, X_data.shape[1]], name='X')
    W = [tf.Variable(each.astype('float32')) for each in W_val]
    _, _, pred = feedforward(X, W[0], W[1], W[2], W[3], W[4], W[5])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred_val = sess.run(pred, {X: X_data})

    return np.argmax(pred_val, axis=1)

def calc_clf_error(X_data, labels, W):
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

def plot_loss(train_loss, val_loss, start=50, per=5, save_file='loss.png'):
    assert len(train_loss) == len(val_loss)
    plt.figure(figsize=(10, 10), facecolor='white')
    idx = np.arange(start, len(train_loss), per)
    plt.plot(idx, train_loss[idx], alpha=1.0, label='train loss')
    plt.plot(idx, val_loss[idx], alpha=1.0, label='val loss')
    plt.xlabel('# of iteration')
    plt.ylabel('loss')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    # plt.show()

def train(args):
    X = load_image(args.input) / 255.
    labels = load_label(args.label)
    Y = dense_to_one_hot(labels, n_class)
    assert X.shape[0] == Y.shape[0]
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
        W, best_loss, train_loss_history, val_loss_history = run_dnn(train_X, train_Y, val_X, val_Y, \
                    args.hidden_dims, lr=args.learning_rate, batch_size=args.batch_size, \
                    max_epoch=args.max_epoch, patience=args.patience, print_per_epoch=args.print_per_epoch)

        print 'best loss: %s' % best_loss

        save_weights(W, args.save_weights)
        print 'saved model weights file to %s' % args.save_weights

        print 'runtime: %ss' % (timeit.default_timer() - start)

        if args.plot_loss:
            plot_loss(train_loss_history, val_loss_history)
        import pdb;pdb.set_trace()

def test(args):
    X = load_image(args.input) / 255.
    labels = load_label(args.label)
    Y = dense_to_one_hot(labels, n_class)
    W = load_weights(args.load_weights)
    assert X.shape[0] == Y.shape[0]

    error_per_class, avg_error = calc_clf_error(X, Y, W)
    print 'error per class: %s' % error_per_class
    print 'average error: %s' % avg_error



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
