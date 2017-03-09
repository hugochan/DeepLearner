import timeit
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_data(file_path, delimiter=' '):

    return np.genfromtxt(file_path, delimiter=delimiter)

def shuffle_data(X_data, Y_data):
    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data = X_data[idx]
    Y_data = Y_data[idx]

    return X_data, Y_data

def calc_mse(X, Y, Theta):
    """Compute the Mean Squared Error.
    """
    a = tf.matmul(X, Theta) - Y
    mse = tf.matmul(a, a, transpose_a=True) / tf.cast(tf.shape(X)[0], tf.float32)

    return mse[0, 0]

def closed_form_grad(X, Y):
    """Compute the closed form gradient-based solution.
    """
    return tf.matmul(tf.matrix_inverse(tf.matmul(X, X, transpose_a=True)), tf.matmul(X, Y, transpose_a=True))

def grad_op(X, Y, Theta, lr=1e-3):
    """Do gradient descent for one iteration.
    """
    grad = 2. * tf.matmul(X, tf.matmul(X, Theta) - Y, transpose_a=True) / tf.cast(tf.shape(X)[0], tf.float32)
    Theta_update = Theta.assign_sub(lr * grad)

    return Theta_update

def next_batch(X, Y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data
        yield (X[i: i + batch_size], Y[i: i + batch_size])

def gd_optimizer(X_data, Y_data, lr=1e-3, max_iter=1000, converge_threshold=1e-5, print_per_iter=10):
    """Gradient descent optimizer.
    """
    X = tf.placeholder(tf.float32, shape=X_data.shape, name='X')
    Y = tf.placeholder(tf.float32, shape=Y_data.shape, name='Y')
    Theta_val = tf.ones((X.get_shape()[1], 1)) * 0.01 # initialize Theta
    Theta = tf.Variable(Theta_val) # initialize Theta

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mse = [] # mse over iterations
        mse_pre = sess.run(calc_mse(X, Y, Theta), {X: X_data, Y: Y_data}) # calc init mse
        mse.append(mse_pre)
        best_theta = Theta_val
        best_mse = mse_pre
        converge = False
        for n_iter in range(max_iter):
            # a) do gradient descent for one iteration
            # b) calc current mse
            Theta_update = grad_op(X, Y, Theta, lr=lr)
            # option 1)
            # Theta_val = sess.run(Theta_update, {X: X_data, Y: Y_data})
            # mse_cur = sess.run(calc_mse(X, Y, Theta), {X: X_data, Y: Y_data})
            # option 2)
            Theta_val, mse_cur = sess.run([Theta_update, calc_mse(X, Y, Theta_update)], {X: X_data, Y: Y_data})
            mse.append(mse_cur)

            if mse_cur < best_mse:
                best_mse = mse_cur
                best_theta = Theta_val

            if n_iter % print_per_iter == 0:
                print 'Iter %s mse: %s' % (n_iter + 1, mse_cur)

            converge = mse_pre - mse_cur < converge_threshold # convergence condition
            if converge:
                print 'Converged in %s iters.' % (n_iter + 1)
                break

            mse_pre = mse_cur

    if not converge:
        print 'Warning: not converged.'

    return best_theta, mse

def sgd_optimizer(X_data, Y_data, lr=1e-3, batch_size=50, max_iter=1000, converge_threshold=1e-5, shuffle=True, print_per_iter=10):
    """Stochastic Gradient descent optimizer.
    """
    X = tf.placeholder(tf.float32, shape=[None, X_data.shape[1]], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, Y_data.shape[1]], name='Y')
    Theta_val = tf.ones((X.get_shape()[1], 1)) * 0.01 # initialize Theta
    Theta = tf.Variable(Theta_val)
    if shuffle:
        X_data, Y_data = shuffle_data(X_data, Y_data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mse = [] # mse over iterations
        mse_pre = sess.run(calc_mse(X, Y, Theta), {X: X_data[0: batch_size], Y: Y_data[0: batch_size]}) # calc init mse
        mse.append(mse_pre)
        best_theta = Theta_val
        best_mse = mse_pre
        n_iter = 1
        while True:
            for (batch_X, batch_Y) in next_batch(X_data, Y_data, batch_size):
                # 1) do gradient descent for one iteration
                # 2) calc current mse
                Theta_update = grad_op(X, Y, Theta, lr=lr)
                Theta_val, mse_batch = sess.run([Theta_update, calc_mse(X, Y, Theta_update)], {X: batch_X, Y: batch_Y})
                mse.append(mse_batch)

                if mse_batch < best_mse:
                    best_mse = mse_batch
                    best_theta = Theta_val

                if n_iter % print_per_iter == 0:
                    print 'Iter %s mse: %s' % (n_iter, mse_batch)

                converge = np.abs(mse_pre - mse_batch) < converge_threshold # convergence condition
                if converge:
                    print 'Converged in %s iters.' % n_iter
                    return best_theta, mse

                if n_iter >= max_iter:
                    print 'Warning: not converged.'
                    return best_theta, mse

                mse_pre = mse_batch
                n_iter += 1

            if shuffle:
                X_data, Y_data = shuffle_data(X_data, Y_data)

    return best_theta, mse

def plot(mse, start=0, per=5):
    idx = np.arange(start, len(mse), per)
    plt.plot(idx, mse[idx])
    plt.xlabel('# of iter')
    plt.ylabel('MSE')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='path to the input data file')
    parser.add_argument('-cf', '--closed_form', action='store_true', help='flag: get the closed form gradient-based solution')
    parser.add_argument('-gd', '--grad_descent', action='store_true', help='flag: use the gradient descent method')
    parser.add_argument('-sgd', '--stochastic_grad_descent', action='store_true', help='flag: use the stochastic gradient descent method')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default 0.001)')
    parser.add_argument('-max_iter', '--max_iteration', type=int, default=1000, help='max number of iterations (default 1000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=50, help='batch size (default 50)')
    parser.add_argument('-conv', '--converge_threshold', type=float, default=1e-8, help='converge threshold (default 1e-5)')
    parser.add_argument('-pp_iter', '--print_per_iter', type=int, default=10, help='print per iteration')
    parser.add_argument('-plot', '--plot_mse', action='store_true', help='flag: plot mse')
    args = parser.parse_args()

    data = load_data(args.input)
    X_data = np.append(data[:, :10], np.ones((data.shape[0], 1)), axis=1).astype('float32')
    Y_data = np.reshape(data[:, -1], (-1, 1)).astype('float32')

    with tf.Session() as sess:
        if args.closed_form:
            start = timeit.default_timer()
            X = tf.Variable(X_data, name='X')
            Y = tf.Variable(Y_data, name='Y')
            Theta = closed_form_grad(X, Y)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print 'The closed form gradient-based solution:'
            print sess.run(Theta)
            print 'runtime: %ss' % (timeit.default_timer() - start)

        if args.grad_descent:
            start = timeit.default_timer()
            Theta, mse = gd_optimizer(X_data, Y_data, \
                    lr=args.learning_rate, max_iter=args.max_iteration, \
                    converge_threshold=args.converge_threshold, \
                    print_per_iter=args.print_per_iter)
            print 'The solution with the gradient descent method:'
            print Theta.reshape((1, -1))
            print 'runtime: %ss' % (timeit.default_timer() - start)
            if args.plot_mse:
                plot(np.array(mse), 2, 1)

        if args.stochastic_grad_descent:
            start = timeit.default_timer()
            Theta, mse = sgd_optimizer(X_data, Y_data, \
                    lr=args.learning_rate, batch_size=args.batch_size, max_iter=args.max_iteration, \
                    converge_threshold=args.converge_threshold, \
                    print_per_iter=args.print_per_iter)
            print 'The solution with the stochastic gradient descent method:'
            print Theta.reshape((1, -1))
            print 'runtime: %ss' % (timeit.default_timer() - start)
            if args.plot_mse:
                plot(np.array(mse), 2, 1)


if __name__ == '__main__':
    main()
