import argparse
import gzip
import os
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from harmonium.optimizers import cd, pcd
from harmonium.rbm import RBMModel, RBMLayer
from harmonium.rbm_utils import save_weights, save_hidden_state
import tensorflow as tf
import urllib.request as request
import sys
from tensorflow.python import debug as tf_debug

from harmonium.regularizers import SparsityTarget


def load_mnist(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = os.path.join(data_path, 'mnist.pkl.gz')

    if not os.path.isfile(dataset):
        print('loading data... ')
        request.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset)
        print('loading data complete... ')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    return [train_set[0], valid_set[0], test_set[0]]


def main():
    parser = argparse.ArgumentParser(
        description='Example of RBM training on MNIST dataset. You can tune some parameters via command line.')

    parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=100, type=int,
                        help='Number of hidden units', required=False)

    parser.add_argument('--visible_bias', action="store_true", dest="visible_bias", default=True,
                        help='Use visible bias', required=False)

    parser.add_argument('--hidden_bias', action="store_true", dest="hidden_bias", default=True,
                        help='Use hidden bias', required=False)

    parser.add_argument('--visible_activation', action="store", dest="visible_activation", default='sigmoid',
                        choices=['sigmoid', 'linear'], help='Visible units type', required=False)

    parser.add_argument('--hidden_activation', action="store", dest="hidden_activation", default='sigmoid',
                        choices=['sigmoid', 'relu'], help='Hidden units type', required=False)

    parser.add_argument('--output_folder', action="store", dest="output_folder", default='./output',
                        help='Folder to store outputs, hidden activations and weights', required=False)

    parser.add_argument('--mnist_folder', action="store", dest="mnist_folder", default='./data',
                        help='Folder containing mnist dataset. If not present, it will be downloaded automatically',
                        required=False)

    parser.add_argument('--tfdebug', action="store", dest="tfdebug", default=None, choices=['cli', 'tensorboard'],
                        help='Use debug session wrapper: cli, tensorboard or none', required=False)

    parser.add_argument('--tbserver', action="store", dest="tbserver", default='localhost:2333',
                        help='TensorBoard server address to use with TensorBoardDebugWrapperSession',
                        required=False)

    params = parser.parse_args(sys.argv[1:])

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    train_set, valid_set, test_set = load_mnist(params.mnist_folder)

    if params.visible_activation == 'linear':
        scaler = StandardScaler()
        train_set = scaler.fit_transform(train_set)
        valid_set = scaler.transform(valid_set)
        test_set = scaler.transform(test_set)

    n_hidden = params.hidden_units
    n_visible = 784

    with tf.Session() as session:
        if params.tfdebug == 'cli':
            session = tf_debug.LocalCLIDebugWrapperSession(session)
        elif params.tfdebug == 'tensorboard':
            session = tf_debug.TensorBoardDebugWrapperSession(session, params.tbserver)

        rbm = RBMModel(visible=RBMLayer(activation=params.visible_activation, units=n_visible,
                                        use_bias=params.visible_bias, sampled=False),
                       hidden=RBMLayer(activation=params.hidden_activation, units=n_hidden,
                                       use_bias=params.hidden_bias, sampled=True),
                       session=session)

        rbm.compile(pcd(1, lr=1e-2), kernel_regularizer=SparsityTarget(l=0.9, p=0.01))

        visualisation_set = valid_set[np.random.randint(len(valid_set), size=400)]
        for i in range(10):
            rbm.fit(train_set, batch_size=128, nb_epoch=10, verbose=2)

            weights = rbm.get_weights()
            save_weights(os.path.join(params.output_folder, 'weights{}.jpg'.format(i)), weights, shape=(28, 28), tile=(10, 10), spacing=(1, 1))
            visualisation_inference = rbm.generate(visualisation_set, sampled=False)

            if params.visible_activation == 'linear':
                visualisation_inference = scaler.inverse_transform(visualisation_inference)

            hidden_state = rbm.hidden_state(visualisation_inference, sampled=True)
            save_weights(os.path.join(params.output_folder, 'output{}.jpg'.format(i)),
                         visualisation_inference.T, shape=(28, 28), tile=(20, 20), spacing=(1, 1))

            save_hidden_state(os.path.join(params.output_folder, 'hidden{}.jpg'.format(i)), hidden_state)
            bias = np.array([0])  # rbm.visible.get_bias()
            print("RBM, epoch {}, GENERATE ERROR: {}, MAX WEIGHT: {}, MIN WEIGHT {}, MEDIAN WEIGHT {}, MAX BIAS {}, MIN BIAS {}, MEDIAN BIAS {}".
                        format(i, mean_squared_error(visualisation_set, visualisation_inference), np.max(weights), np.min(weights),
                               np.median(weights), np.max(bias), np.min(bias), np.median(bias)))


if __name__ == "__main__":
    sys.exit(main())
