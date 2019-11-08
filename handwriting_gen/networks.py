import os
import pickle
from pathlib import Path
import tensorflow as tf


def create_tf_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def make_lstm_cell(hidden_size, dropout=0.0, train=True):
    """
    Create LSTM cell with optional dropout
    """
    lstm_cell = tf.nn.rnn_cell.LSTMCell(
        hidden_size, state_is_tuple=True)
    if train and dropout > 0:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=1 - dropout)
    return lstm_cell


def get_lstm_cells_and_states(hidden_size, train, num_layers,
                              batch_size, dropout):
    cells = [make_lstm_cell(hidden_size, dropout, train)
             for _ in range(num_layers)]
    initial_states = [
        c.zero_state(batch_size=batch_size, dtype=tf.float32)
        for c in cells]
    return cells, initial_states


def stacked_lstm_model(inputs, hidden_size, num_layers,
                       batch_size, dropout=0, train=True):
    """
    Create graph for stacked LSTM manually
        tf.nn.rnn_cell.MultiRNNCell was
        causing undesirable results in training with tensorflow==1.8

    :param inputs: sequence of inputs for each RNN step
    :param hidden_size: int, number of hidden units
    :param num_layers: int, number of layers to stack
    :param batch_size: int, number of rows of input
        (necessary to get initial_state)
    :param dropout: float, dropout probability
    :param train: bool, if drouput > 0 and train = True, then we add dropout
    """
    assert num_layers > 0, "Num layers must be > 0"

    cells, initial_states = get_lstm_cells_and_states(
        hidden_size, train, num_layers, batch_size, dropout)

    outputs = []
    last_states = []
    last_inputs = inputs
    for idx, (i, c) in enumerate(zip(initial_states, cells)):

        out, last_state = tf.nn.static_rnn(
            c, last_inputs, initial_state=i, dtype=tf.float32,
            scope='layer{}'.format(idx))

        outputs.append(out)
        last_states.append(last_state)
        last_inputs = out

    return outputs, last_states, initial_states


def get_embedding(inputs, vocab_size, hidden_size,
                  initializer=tf.glorot_normal_initializer()):
    """
    word/character embedding
    :param inputs: tf.placeholder
    :param vocab_size: int, size of vocab
    :param hidden_size: int
    :param initializer: weights initializer
    """
    embedding = tf.get_variable(
        "embedding", [vocab_size, hidden_size], dtype=tf.float32,
        initializer=initializer)
    return tf.nn.embedding_lookup(embedding, inputs)


class TFModel:
    """
    Base tensorflow model for saving/loading
    """

    def __init__(self, save_path):
        self.save_path = save_path
        Path(save_path).mkdir(exist_ok=True)

    @classmethod
    def load(cls, save_path, **kwargs):
        tf.reset_default_graph()

        with open(os.path.join(save_path, 'test.pkl'), 'rb') as f:
            params = pickle.load(f)

        params.update(kwargs)
        params['save_path'] = save_path

        print('Loading: ', params)

        self = cls(**params)
        self.saver.restore(
            self.sess, tf.train.latest_checkpoint(self.save_path))
        return self

    def save(self):
        serializables = []
        for k, v in self.__dict__.items():
            if type(v) in [dict, int, float, bool, str]:
                serializables.append((k, v))
        serializables = dict(serializables)
        f = os.path.join(self.save_path, 'test.pkl')
        with open(f, 'wb') as f:
            pickle.dump(serializables, f)
        self.saver.save(self.sess, self.save_path,
                        tf.train.get_or_create_global_step())
