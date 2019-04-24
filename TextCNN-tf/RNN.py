import tensorflow as tf
import numpy as np

class RNN:

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 cell_type, hidden_size, l2_reg_lambda=0.1, pretrain_enable=True):

        # Placeholders for input, output and dropout

        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size],
                                                    name="embedding_placeholder")
        # Keeping track of l2 regularization loss (optional)

        l2_loss = tf.constant(0.0)

        # Embedding layer

        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            if pretrain_enable:
                W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), name="W")
                self.embedding_init = tf.assign(W, self.embedding_placeholder, name="embedding_init")
            else:
                W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedding_init = tf.constant(0.0) # Disable the Operation for Pretrain Vector

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        # Recurrent Neural Network

        with tf.name_scope("rnn"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            rnn_out, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_chars, dtype=tf.float32)
            self.last_out = rnn_out[:, -1, :]

        # Final scores and predictions

        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[hidden_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.last_out, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")