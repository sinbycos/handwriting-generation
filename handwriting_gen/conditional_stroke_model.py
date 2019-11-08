"""
Conditional stroke generation
"""
import os
import click
import numpy as np
import tensorflow as tf
from handwriting_gen.data_manager import (
    get_preprocessed_data_splits, conditional_batch_generator,
    sent_to_int, pad_vec
)
from handwriting_gen import plotting
from handwriting_gen.networks import TFModel, get_lstm_cells_and_states
from handwriting_gen.networks import create_tf_session
from handwriting_gen.distributions import build_stroke_mixture_model
from handwriting_gen.distributions import build_window_mixture_model
from handwriting_gen.sample import sample_stroke

STROKE_DIM = 3  # (end-of-stroke, x, y) - input dimension


class ConditionalStrokeModel(TFModel):

    def __init__(self, save_path,
                 learning_rate=1e-4,
                 mixture_components=20,
                 window_components=10,
                 char_dict={},
                 char_seq_len=30,
                 hidden_size=200, dropout=0.0,
                 max_grad_norm=10, decay=0.95, momentum=0.9,
                 rnn_steps=200, batch_size=32, is_train=True, **kwargs):
        """
        :param save_path: str, path to model checkpoints
        :param learning_rate: float
        :param mixture_components: int,
            number of mixture components to predict strokes
        :param window_components: int, number of components for window model
        :param char_dict: dictionary of chars to ints
        :param char_seq_len: int, length of character sequence
        :param hidden_size: int, number of hidden units for lstm
        :param dropout: float, prob for output dropout
        :param max_grad_norm: int, gradient clipping value
        :param decay: float, decay for RMSProp
        :param momentum: float, momentutm for RMSProp
        :param rnn_steps: int, steps to rollout RNN
        :param batch_size: int, size of batch for training
        :param is_train: bool, whether we are training
            or evaluating the network
        """
        super().__init__(save_path)

        self.sess = create_tf_session()
        self.learning_rate = learning_rate
        self.mixture_components = mixture_components
        self.vocab_size = len(char_dict)
        self.char_dict = char_dict
        self.window_components = window_components
        self.char_seq_len = char_seq_len
        self.num_layers = 3
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.max_grad_norm = max_grad_norm
        self.decay = decay
        self.momentum = momentum
        self.rnn_steps = rnn_steps
        self.batch_size = batch_size
        self.is_train = is_train

        self.build()
        self.sess.run(tf.global_variables_initializer())

    def build(self):

        # the input is the stroke sequence
        self.inputs = inputs = tf.placeholder(
            tf.float32, [None, self.rnn_steps, STROKE_DIM], 'input')
        # the target is the next stroke
        self.targets = tf.placeholder(
            tf.float32, [None, self.rnn_steps, STROKE_DIM], 'targets')
        # the input characters for the sequence
        self.input_characters = tf.placeholder(
            tf.int32, [None, self.char_seq_len],
            'characters')
        self.characters = tf.one_hot(
            self.input_characters, depth=self.vocab_size)

        with tf.variable_scope('char-stroke-gen', reuse=tf.AUTO_REUSE):
            # split inputs by rnn steps
            inputs = [tf.squeeze(i, [1]) for i in
                      tf.split(self.inputs, self.rnn_steps, axis=1)]

            cells, initial_states = get_lstm_cells_and_states(
                self.hidden_size, self.is_train, self.num_layers,
                self.batch_size, self.dropout)
            self.initial_states = initial_states

            last_outputs = build_window_mixture_model(
                self, initial_states, cells, inputs,
                self.batch_size, self.window_components,
                self.characters, self.char_seq_len)

            output = tf.reshape(
                tf.concat(last_outputs, 1), [-1, self.hidden_size])

            build_stroke_mixture_model(
                self, output, self.targets, self.mixture_components,
                self.max_grad_norm, self.learning_rate, self.decay,
                self.momentum)

        self.saver = tf.train.Saver()

    def train(self, inputs, targets, chars):
        return self.sess.run(
            [self.loss, self.train_op],
            {self.inputs: inputs, self.targets: targets,
             self.input_characters: chars})


def decode(model, text='hey I am Tanushri, nice to meet you',
           seed=42, std_bias=10, mixture_bias=1):
    """
    Decode strokes from network conditional on text input
    """

    np.random.seed(seed)

    text_int = sent_to_int(list(text), model.char_dict)
    text_int = pad_vec([text_int], model.char_seq_len)[0]

    feed_dict = {}
    feed_dict[model.input_characters] = [text_int]

    strokes = [np.array([[[1, 0, 0]]])]
    last_state, last_kappa, last_wt = model.sess.run(
        [model.initial_states, model.init_kappa, model.init_wt],
        feed_dict)

    timestep = 0
    while timestep < 700:
        timestep += 1

        feed_dict[model.inputs] = strokes[-1]
        for idx, i in enumerate(model.initial_states):
            feed_dict[i] = last_state[idx]
        feed_dict[model.init_kappa] = last_kappa

        # setting last_wt matters if you feed it into the first LSTM layer
        # in the network
        feed_dict[model.init_wt] = last_wt

        last_state, last_wt, last_kappa, phi, (e, pi, *bv_params) = model.sess.run(
            [model.last_states, model.last_wt, model.last_kappa, model.phi,
             model.bivariate_normal_params],
            feed_dict)
        phi = phi.squeeze()

        # sample mixture params
        s = sample_stroke(pi, e, mixture_bias, bv_params, std_bias)
        strokes.append(s)

        # stopping heuristic
        if np.all(phi[-1] > phi[:-1]):
            break

    # append a pen lift at the end so that
    #   plotting works at the beginning of training
    strokes[-1] = np.array([[[1, 0, 0]]])
    strokes = np.array(strokes).squeeze()
    return strokes


def _print_loss(model, data, epoch, stroke_length, char_length,
                batch_size, description=''):
    epoch_size = len(data[0]) // batch_size
    losses = []
    data_gen = conditional_batch_generator(
        data, stroke_length=stroke_length,
        char_length=char_length, batch_size=batch_size)
    for i in range(epoch_size):
        input_batch, target_batch, char_batch = next(data_gen)
        loss, mse = model.sess.run(
            [model.loss, model.mse],
            {model.inputs: input_batch, model.targets: target_batch,
             model.input_characters: char_batch})
        losses.append(loss)

    print('Epoch: {}, {} loss: {}'.format(
        epoch, description, np.mean(losses)))


@click.command()
@click.argument('model-folder')
@click.option('--num-epochs', default=3)
@click.option('--learning-rate', default=1e-4)
@click.option('--batch-size', default=32)
@click.option('--stroke-length', default=150)
@click.option('--steps-per-char', default=22,
              help='the avg number of stroke steps per character')
@click.option('--save-every', default=20)
def train(model_folder, num_epochs, learning_rate, batch_size,
          stroke_length, steps_per_char, save_every):

    char_seq_length = stroke_length // steps_per_char

    train_data, val_data, test_data, metadata = get_preprocessed_data_splits()
    train_data_gen = conditional_batch_generator(
        train_data, stroke_length=stroke_length,
        char_length=char_seq_length, batch_size=batch_size)
    epoch_size = len(train_data[0]) // batch_size

    model = ConditionalStrokeModel(
        model_folder, learning_rate=learning_rate,
        batch_size=batch_size, rnn_steps=stroke_length,
        is_train=True, char_dict=metadata['char_dict'],
        char_seq_len=char_seq_length)

    for epoch in range(num_epochs):
        for i in range(epoch_size):
            input_batch, target_batch, char_batch = next(train_data_gen)
            loss, _ = model.train(input_batch, target_batch, char_batch)

        if not epoch % save_every and epoch != 0:
            model.save()

        if not epoch % save_every and epoch != 0:
            model = ConditionalStrokeModel.load(
                model_folder, batch_size=1,
                rnn_steps=1, is_train=False,
                char_seq_len=30)
            strokes = decode(model)

            plotting.plot_stroke(
                strokes, os.path.join(model_folder, 'test{}'.format(epoch)))

            model = ConditionalStrokeModel.load(
                model_folder, learning_rate=learning_rate,
                batch_size=batch_size, rnn_steps=stroke_length,
                is_train=True,
                char_seq_len=char_seq_length)

        _print_loss(model, train_data, epoch,
                    stroke_length, char_seq_length, batch_size, 'Train')
        _print_loss(model, val_data, epoch,
                    stroke_length, char_seq_length, batch_size, '    Val')
    model.save()


if __name__ == '__main__':
    train()
