"""
Model training script to unconditionally generate handwritten strokes.

Network architecture has less hidden units compared to [1]
and we don't add skip connections for simplicity.

[1] https://arxiv.org/pdf/1308.0850.pdf
"""

import os
import click
import numpy as np
import tensorflow as tf
from handwriting_gen.data_manager import (
    get_preprocessed_data_splits, stroke_batch_generator
)
from handwriting_gen import plotting
from handwriting_gen.networks import (
    stacked_lstm_model, TFModel, create_tf_session)
from handwriting_gen.distributions import build_stroke_mixture_model
from handwriting_gen.sample import sample_stroke

STROKE_DIM = 3  # (end-of-stroke, x, y) - input dimension


class UnconditionalStrokeModel(TFModel):

    def __init__(self, save_path,
                 learning_rate=1e-4,
                 mixture_components=20,
                 num_layers=3, hidden_size=200, dropout=0.0,
                 max_grad_norm=5, decay=0.95, momentum=0.9,
                 rnn_steps=200, batch_size=32, is_train=True, **kwargs):
        """
        :param save_path: str, path to model checkpoints
        :param learning_rate: float
        :param mixture_components: int,
            number of mixture components to predict strokes
        :param num_layers: int, number of layers for stacked lstm
        :param hidden_size: int, number of hidden units for lstm
        :param dropout: float, prob for output dropout
        :param max_grad_norm: int, gradient clipping value
        :param decay: float, decay for RMSProp
        :param momentum: float, momentutm for RMSProp
        :param rnn_steps: int, steps to rollout RNN
        :param batch_size: int, size of batch for training
        :param train: bool, whether we are training or evaluating the network

        paper defaults to 3 layer LSTM with 400 hidden units
        output derivatives were clipped to -100, 100 and lstm derivs to -10, 10
        but we'll just do 200 hidden units and +/-5 clippingfor now. we also
        don't implement the skip connections for convenience.
        """
        super().__init__(save_path)

        self.sess = create_tf_session()

        self.learning_rate = learning_rate
        self.mixture_components = mixture_components
        self.num_layers = num_layers
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

        self.inputs = inputs = tf.placeholder(
            tf.float32, [None, self.rnn_steps, STROKE_DIM],
            'input')
        self.targets = tf.placeholder(
            tf.float32, [None, self.rnn_steps, STROKE_DIM],
            'targets')

        with tf.variable_scope('stroke-gen', reuse=tf.AUTO_REUSE):
            # split inputs by rnn steps
            inputs = [tf.squeeze(i, [1]) for i in
                      tf.split(self.inputs, self.rnn_steps, axis=1)]

            outputs, last_state, initial_state = stacked_lstm_model(
                inputs, self.hidden_size, self.num_layers,
                self.batch_size, self.dropout, self.is_train)
            self.initial_state, self.last_state = initial_state, last_state

            output = tf.reshape(
                tf.concat(outputs[-1], 1), [-1, self.hidden_size])

            build_stroke_mixture_model(
                self, output, self.targets, self.mixture_components,
                self.max_grad_norm, self.learning_rate, self.decay,
                self.momentum)

        self.saver = tf.train.Saver()

    def train(self, inputs, targets):
        return self.sess.run(
            [self.loss, self.train_op],
            {self.inputs: inputs, self.targets: targets})


def decode(model, steps=650, seed=42, std_bias=10, mixture_bias=1):
    """
    Decode strokes from network
    :param model: model with 1 batch_size and 1 rnn_step
    :param steps: int, steps to rollout
    :param seed: int, random seed
    :param std_bias: float, denominator on standard deviations
        higher value makes us track closer to the mean
    :param mixture_bias: float, higher values make us more greedy in terms
        of selecting a mixture to track
    """

    np.random.seed(seed)

    strokes = [np.array([[[1, 0, 0]]])]
    last_state = model.sess.run(model.initial_state)

    for _ in range(steps):

        feed_dict = {model.inputs: strokes[-1]}
        for idx, i in enumerate(model.initial_state):
            feed_dict[i] = last_state[idx]

        last_state, (e, pi, *bv_params) = model.sess.run(
            [model.last_state, model.bivariate_normal_params],
            feed_dict)

        s = sample_stroke(pi, e, mixture_bias, bv_params, std_bias)
        strokes.append(s)

    # append a pen lift at the end so that plotting works during early training
    strokes[-1] = np.array([[[1, 0, 0]]])
    strokes = np.array(strokes).squeeze()
    return strokes


def _print_loss(model, data, epoch, stroke_length, batch_size, description=''):
    epoch_size = len(data[0]) // batch_size
    losses, mses = [], []
    data_gen = stroke_batch_generator(
        data, length=stroke_length, batch_size=batch_size)
    for i in range(epoch_size):
        input_batch, target_batch = next(data_gen)
        loss, mse = model.sess.run(
            [model.loss, model.mse],
            {model.inputs: input_batch, model.targets: target_batch})
        losses.append(loss)
        mses.append(mse)
    print('Epoch: {}, {} loss: {}, mse: {}'.format(
        epoch, description, np.mean(losses), np.mean(mses)))


@click.command()
@click.argument('model-folder',)
@click.option('--num-epochs', default=3)
@click.option('--learning-rate', default=1e-4)
@click.option('--batch-size', default=32)
@click.option('--stroke-length', default=200)
@click.option('--save-every', default=20)
def train(model_folder, num_epochs, learning_rate, batch_size,
          stroke_length, save_every):

    train_data, val_data, test_data, metadata = get_preprocessed_data_splits()
    train_data_gen = stroke_batch_generator(
        train_data, length=stroke_length, batch_size=batch_size)

    model = UnconditionalStrokeModel(
        model_folder, learning_rate=learning_rate,
        batch_size=batch_size, rnn_steps=stroke_length,
        is_train=True)
    epoch_size = len(train_data[0]) // batch_size

    for epoch in range(num_epochs):
        for i in range(epoch_size):
            input_batch, target_batch = next(train_data_gen)
            model.train(input_batch, target_batch)

        if not epoch % save_every and epoch != 0:
            model.save()

        if not epoch % save_every and epoch != 0:
            # make some stroke plots
            model = UnconditionalStrokeModel.load(
                model_folder, batch_size=1,
                rnn_steps=1, is_train=False)

            strokes = decode(model, 650, epoch)
            plotting.plot_stroke(
                strokes, os.path.join(model_folder, 'test{}'.format(epoch)))

            model = UnconditionalStrokeModel.load(
                model_folder, learning_rate=learning_rate,
                batch_size=batch_size, rnn_steps=stroke_length)

        _print_loss(model, train_data, epoch,
                    stroke_length, batch_size, 'Train')
        _print_loss(model, val_data, epoch,
                    stroke_length, batch_size, '    Val')
    model.save()


if __name__ == '__main__':
    train()
