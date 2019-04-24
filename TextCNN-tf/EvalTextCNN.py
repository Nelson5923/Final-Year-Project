import tensorflow as tf
import numpy as np
import os
import time
import datetime
import DataPreprocessing
from TextCNN import TextCNN
from tensorflow.contrib import learn
import csv

# Eval Parameters

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("pretrain_enable", True, "Add word2vec pretrain vector")

FLAGS = tf.flags.FLAGS
# CHANGE THIS: Load data. Load your own data here

x_raw, y_raw, embedding, _ = DataPreprocessing.Preprocessor("./data/TestCorpus.txt", "./data/TestLabel.txt", "./word2vec.model")
y_raw = np.argmax(y_raw, axis=1)

print("\nEvaluating...\n")

# Evaluation
# ==================================================

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():

    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)

    with sess.as_default():

        # Load the saved meta graph and restore variables

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        embedding_placeholder = graph.get_operation_by_name("embedding_placeholder").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate

        if FLAGS.pretrain_enable is True:
            embedding_init = graph.get_operation_by_name("embedding/embedding_init").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        else:
            embedding_init = tf.constant(0.0)
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch

        batches = DataPreprocessing.BatchIterator(list(x_raw), FLAGS.batch_size, 1, Shuffle=False)

        # Collect the predictions here

        predicted = []

        for batch in batches:

            feed_dict = {
                input_x: batch,
                dropout_keep_prob: 1.0,
                embedding_placeholder: embedding
            }

            _, result = sess.run([embedding_init, predictions], feed_dict)
            predicted = np.concatenate([predicted, result])

# Result Summary

actual = y_raw

if actual is not None:

    print(predicted)
    print(actual)
    TP = np.count_nonzero(predicted * actual)
    TN = np.count_nonzero((predicted - 1) * (actual - 1))
    FP = np.count_nonzero(predicted * (actual - 1))
    FN = np.count_nonzero((predicted - 1) * actual)
    accuracy = (TP + TN) / (TP + FN + TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    print("Total Spam: {}/{}".format(sum(actual), len(actual)))
    print("Total Predicted Spam: {}/{}".format(sum(predicted), len(predicted)))
    print("TP: {} TN: {}".format(TP, TN))
    print("FP: {} FN: {}".format(FP, FN))
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))

# Save the evaluation to a csv
'''
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
'''