import tensorflow as tf
import numpy as np
import os
import time
import datetime
import DataPreprocessing
from RNN import RNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters

tf.flags.DEFINE_integer("embedding_dim", 250, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 250, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.3, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (Default: 3.0)")

# Training parameters

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("pretrain_enable", True, "Add word2vec pretrain vector")
tf.flags.DEFINE_boolean("cell_type", True, "Add word2vec pretrain vector")

FLAGS = tf.flags.FLAGS

def train(x_train, y_train, x_dev, y_dev, embedding, vocab_processor):

    # Training
    # ==================================================

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)

        with sess.as_default():

            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrain_enable=FLAGS.pretrain_enable
            )

            # Define Training procedure

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy

            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries

            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries

            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary

            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables

            sess.run(tf.global_variables_initializer())

            # Single Training Step

            def train_step(x_batch, y_batch, embedding):

                feed_dict = {
                  rnn.input_x: x_batch,
                  rnn.input_y: y_batch,
                  rnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                  rnn.embedding_placeholder: embedding
                }

                _, _, step, summaries, loss, accuracy, prediction, la_out = sess.run(
                    [rnn.embedding_init, train_op, global_step, train_summary_op,
                     rnn.loss, rnn.accuracy, rnn.predictions, rnn.last_out],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                print("Total Expected Spam :{}/{}".format(sum(prediction), len(prediction)))
                # print("Expected Score :{}".format(score))
                # print("Embedding Matrix :{}".format(p_emb))
                print("RNN :{}".format(la_out.shape))
                train_summary_writer.add_summary(summaries, step)

            # Model Evaluation

            def dev_step(x_batch, y_batch, embedding, writer=None):

                feed_dict = {
                  rnn.input_x: x_batch,
                  rnn.input_y: y_batch,
                  rnn.dropout_keep_prob: 1.0,
                  rnn.embedding_placeholder: embedding
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches

            batches = DataPreprocessing.BatchIterator(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, embedding)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, embedding, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                if current_step == 20000:
                    break;

            sess.close();

def main(argv=None):

    x_raw, y_raw, embedding, vocab_processor = DataPreprocessing.Preprocessor \
        ("./data/TrainCorpus.txt", "./data/TrainLabel.txt", "./word2vec.model")
    x_train, y_train,  x_dev, y_dev = DataPreprocessing.DataSplit(x_raw, y_raw, 0.01)

    start_time = time.time()

    train(x_train, y_train, x_dev, y_dev, embedding, vocab_processor)

    end_time = time.time()
    time_dif = end_time - start_time
    print("Total Running Time:" + str(time_dif))


if __name__ == '__main__':

    tf.app.run()



