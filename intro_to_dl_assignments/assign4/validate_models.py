#!/usr/bin/python

import argparse
try:
  import cPickle as pickle
except ImportError:
  # Python 3
  import pickle
import os
import sys

import numpy as np

import tensorflow as tf


# How many images to include in each validation batch. This is just a default
# value, and may be set differently to accomodate network parameters.
batch_size = 1000


def extract_validation_handles(session):
  """ Extracts the input and predict_op handles that we use for validation.
  Args:
    session: The session with the loaded graph.
  Returns:
    The inputs placeholder and the prediction operation. """
  # The students should have saved their input placeholder and prediction
  # operation in a collection called "validation_nodes".
  valid_nodes = tf.get_collection_ref("validation_nodes")
  if len(valid_nodes) != 2:
    print("ERROR: Expected 2 items in validation_nodes, got %d." % \
          (len(valid_nodes)))
    sys.exit(1)

  # Figure out which is which.
  inputs = valid_nodes[0]
  predict = valid_nodes[1]
  if type(valid_nodes[1]) == tf.placeholder:
    inputs = valid_nodes[1]
    predict = valid_nodes[0]

  # Check to make sure we've set the batch size correctly.
  try:
    global batch_size
    batch_size = int(inputs.get_shape()[0])
    print("WARNING: Network does not support variable batch sizes.")
  except TypeError:
    # It's unspecified, which is actually correct.
    pass

  # Predict op should also yield integers.
  predict = tf.cast(predict, "int32")

  # Check the shape of the prediction output.
  p_shape = predict.get_shape()
  if len(p_shape) > 2:
    print("ERROR: Expected prediction of shape (<X>, 1), got shape of %s." % \
          (str(p_shape)))
    sys.exit(1)
  if len(p_shape) == 2:
    if p_shape[1] != 1:
      print("ERROR: Expected prediction of shape (<X>, 1), got shape of %s." % \
            (str(p_shape)))
      sys.exit(1)

    # We need to contract it into a vector.
    predict = predict[:, 0]

  return (inputs, predict)

def load_model(session, save_path):
  """ Loads a saved TF model from a file.
  Args:
    session: The tf.Session to use.
    save_path: The save path for the saved session, returned by Saver.save().
  Returns:
    The inputs placehoder and the prediction operation.
  """
  print("Loading model from file '%s'..." % (save_path))

  meta_file = save_path + ".meta"
  if not os.path.exists(meta_file):
    print("ERROR: Expected .meta file '%s', but could not find it." % \
          (meta_file))
    sys.exit(1)

  saver = tf.train.import_meta_graph(meta_file)
  # It's finicky about the save path.
  save_path = os.path.join("./", save_path)
  saver.restore(session, save_path)
  # Check that we have the handles we expected.
  return extract_validation_handles(session)

def load_validation_data(val_filename):
  """ Loads the validation data.
  Args:
    val_filename: The file where the validation data is stored.
  Returns:
    A tuple of the loaded validation data and validation labels. """
  print("Loading validation data...")

  val_data_file = file(val_filename, "rb")
  val_x, val_y = pickle.load(val_data_file)
  val_data_file.close()

  # Convert to floats.
  val_x = val_x.astype("float32")
  # Scale.
  val_x /= 255.0
  # Subtract the mean.
  val_x -= np.mean(val_x)

  return (val_x, val_y)

def validate_model(session, val_data, inputs, predict_op):
  """ Validates the model stored in a session.
  Args:
    session: The session where the model is loaded.
    val_data: The validation data to use for evaluating the model.
    inputs: The inputs placeholder.
    predict_op: The prediction operation.
  Returns:
    The overall validation accuracy for the model. """
  print("Validating model...")

  # Extend the graph to count the number of items that match up with the
  # validation labels.
  valid_labels = tf.placeholder("int32")
  correct = tf.equal(predict_op, valid_labels)
  # Compute total number of correct answers.
  total_correct = tf.reduce_sum(tf.cast(correct, "float32"))

  # Validate the model.
  val_x, val_y = val_data
  num_iters = val_x.shape[0] / batch_size
  all_correct = 0
  for i in range(0, num_iters):
    start_index = i * batch_size
    end_index = start_index + batch_size
    val_batch = val_x[start_index:end_index]
    label_batch = val_y[start_index:end_index]

    print("Validating batch %d of %d..." % (i + 1, num_iters))
    all_correct += session.run(total_correct, feed_dict={inputs: val_batch,
                               valid_labels: label_batch})

  # Compute total accuracy.
  accuracy = all_correct / val_x.shape[0]
  return accuracy


def main():
  parser = argparse.ArgumentParser("Analyze student models.")
  parser.add_argument("-v", "--val_data_file", default=None,
                      help="Validate the network with the data from this " + \
                           "pickle file.")
  parser.add_argument("save_path", help="The base path for your saved model.")
  args = parser.parse_args()

  if not args.val_data_file:
    print("Not validating, but checking network compatibility...")
  elif not os.path.exists(args.val_data_file):
    print("ERROR: Could not find validation data '%s'." % (args.val_data))
    sys.exit(1)

  # Load and validate the network.
  with tf.Session() as session:
    inputs, predict_op = load_model(session, args.save_path)
    if args.val_data_file:
      val_data = load_validation_data(args.val_data_file)
      accuracy = validate_model(session, val_data, inputs, predict_op)

      print("Overall validation accuracy: %f" % (accuracy))

    else:
      print("Network seems good. Go ahead and submit.")

if __name__ == "__main__":
  main()

