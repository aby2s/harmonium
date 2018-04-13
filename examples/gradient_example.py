import gzip
import os
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from boltzmann.optimizers import cd, pcd
import boltzmann.regularizers as regularizers
from boltzmann.rbm import RBMModel, RBMLayer
from boltzmann.rbm_utils import save_weights, save_hidden_state
import tensorflow as tf
import urllib.request as request

from boltzmann.regularizers import SparsityTarget


train_set = np.array([[1,0,1,0],[1,0,1,1]])

n_hidden = 2
n_visible = 4

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

rbm = RBMModel(visible=RBMLayer(activation='linear', units=n_visible, use_bias=True, sampled=False),
               hidden=RBMLayer(activation='sigmoid', units=n_hidden, use_bias=True, sampled=True))
rbm.compile(cd(1, lr=1e-3))#, kernel_regularizer=regularizers.SparsityTarget(0.01, 0.5), bias_regularizer=regularizers.L2(0.1))
for i in range(1):
    rbm.fit(train_set, batch_size=2, nb_epoch=2, verbose=2, trace=True)
    with open('reses.pickle', 'wb') as f:
        pickle.dump(rbm.trace_data, f)
    weights = rbm.get_weights()

