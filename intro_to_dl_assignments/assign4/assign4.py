import argparse
import cPickle as pickle
import numpy as np
import tensorflow as tf

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

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k, strides):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                          padding='VALID')

# Create model
def conv_net(x, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, img_size, img_size, n_channel])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=conv1_stride)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=pool1_size, strides=pool1_stride)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides=conv2_stride)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=pool2_size, strides=pool2_stride)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=conv3_stride)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, fc_size])
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def run_cnn(train_x, train_y, val_x, val_y, lr=1e-3, batch_size=128, max_epoch=1000, min_delta=1e-4, patience=10, print_per_epoch=10, out_model='my_model'):
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    # Store layers weight & bias
    weights = {
        'wc1': tf.get_variable('wc1', shape=(conv1_size, conv1_size, n_channel, conv1_out), initializer=tf.contrib.layers.xavier_initializer()),
        'wc2': tf.get_variable('wc2', shape=(conv2_size, conv2_size, conv1_out, conv2_out), initializer=tf.contrib.layers.xavier_initializer()),
        'wc3': tf.get_variable('wc3', shape=(conv3_size, conv3_size, conv2_out, conv3_out), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('out', shape=(fc_size, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
    }

    biases = {
        'bc1': tf.Variable(tf.constant(0.1, shape=[conv1_out])),
        'bc2': tf.Variable(tf.constant(0.1, shape=[conv2_out])),
        'bc3': tf.Variable(tf.constant(0.1, shape=[conv3_out])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    # correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    train_loss_history = []
    val_loss_history = []
    n_incr_error = 0  # nb. of consecutive increase in error
    best_loss = np.Inf
    n_batches = train_x.shape[0] / batch_size + (train_x.shape[0] % batch_size != 0)

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
            train_x, train_y = shuffle_data(train_x, train_y)
            for batch_x, batch_y in next_batch(train_x, train_y, batch_size):
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                train_batch_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                train_loss += train_batch_loss / n_batches
                val_batch_loss = sess.run(cost, feed_dict={x: val_x, y: val_y})
                val_loss += val_batch_loss / n_batches

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            current_loss = val_loss
            if current_loss - min_delta < best_loss:
                # update best error (NLL), iteration
                best_loss = current_loss
                n_incr_error = 0

            if n_epoch % print_per_epoch == 0:
                print 'Epoch %s/%s, train loss: %s, val loss: %s' % (n_epoch, max_epoch, train_loss, val_loss)

            if n_incr_error >= patience:
                print 'Early stopping occured. Optimization Finished!'
                save_model(sess, x, tf.argmax(pred, 1), out=out_model)
                return train_loss_history, val_loss_history

        # # Calculate accuracy for 256 mnist test images
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
        #                                   y: mnist.test.labels[:256]}))
        save_model(sess, x, tf.argmax(pred, 1), out=out_model)
        return train_loss_history, val_loss_history

def save_model(sess, inputs, predict_op, out='my_model'):
    # Create the collection.
    tf.get_collection("validation_nodes")
    # Add stuff to the collection.
    tf.add_to_collection("validation_nodes", inputs)
    tf.add_to_collection("validation_nodes", predict_op)
    saver = tf.train.Saver()
    save_path = saver.save(sess, out)

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

    train_loss_hist, test_loss_hist = run_cnn(train_x, train_y, test_x, test_y, \
                                            lr=args.learning_rate, \
                                            batch_size=args.batch_size, \
                                            max_epoch=args.max_epoch, \
                                            min_delta=1e-4, \
                                            patience=args.patience, \
                                            print_per_epoch=args.print_per_epoch,
                                            out_model=args.save_model)

    import pdb;pdb.set_trace()
    plot_loss(train_loss_hist, test_loss_hist, start=0, per=1, save_file='loss.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the input data')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations (default 100)')
    # parser.add_argument('-n_val', '--num_validation', type=int, default=100, help='validation set size (default 100)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size (default 50)')
    parser.add_argument('-p', '--patience', type=int, default=10, help='early stopping patience (default 10)')
    parser.add_argument('-pp_iter', '--print_per_epoch', type=int, default=1, help='print per iteration (default 10)')
    parser.add_argument('-sm', '--save_model', type=str, default='my_model', help='path to the output model (default my_model)')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the loaded model')
    # parser.add_argument('-pw', '--plot_weights', action='store_true', help='flag: plot weights')
    # parser.add_argument('-pl', '--plot_loss', action='store_true', help='flag: plot loss')
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        if not args.load_model:
            raise Exception('load_model arg needed in test phase')
        test(args)

if __name__ == '__main__':
    main()
