import gzip
import os
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error

from boltzmann.rbm import RBMModel, cd, RBMLayer, l2, l1
from boltzmann.rbm_utils import save_weights, save_hidden_state
import tensorflow as tf
import urllib.request as request


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


train_set, valid_set, test_set = load_mnist('.//data')

n_hidden = 100
n_visible = 784

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

rbm = RBMModel(visible=RBMLayer(activation='linear', units=n_visible, use_bias=True, sampled=False), hidden=RBMLayer(activation='sigmoid', units=n_hidden, use_bias=True, sampled=True))
rbm.compile(cd(1, lr=1e-2), unstack=True, kernel_regularizer=l1(0.00001))
for i in range(10):
    rbm.fit(train_set, batch_size=256, nb_epoch=2, verbose=1)
    #bias = rbm.visible.get_bias()
    weights = rbm.get_weights()
    save_weights('weights/weights{}.jpg'.format(i), weights, shape=(28, 28), tile=(10, 10), spacing=(1,1))
    for j in range(20):
        batch = train_set[j * 200: (j + 1) * 200]
        batch_t = rbm.generate(batch, sampled=False)
        hidden_state = rbm.hidden_state(batch, 1, sampled=False)

        save_weights('class_out/output{}.jpg'.format(j), batch_t.T, shape=(28, 28), tile=(20, 10),
                     spacing=(1, 1))
        save_hidden_state('class_out/hidden{}.jpg'.format(j), hidden_state)
        print("RBM, epoch {}, GENERATE ERROR: {}, MAX WEIGHT: {}, MIN WEIGHT {}, MEDIAN WEIGHT {}".
              format(i, mean_squared_error(batch, batch_t), np.max(weights), np.min(weights),
                     np.median(weights)))