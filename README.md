# Harmonium
Simple Restricted Boltzmann Machine implementation with TensorFlow.
Implementation supports sigmoid, relu and linear visible/hidden units, l1/l2/sparsity regularization, contrastive divergence and stochastic maximum likelihood learning (sml/pcd) with momentum.



# Writing a Simple RBM

```python
 # Here is a simple code sample
 # See harmonium/rbm.py for an API documentation
 # You can also find full MNIST example under examples folder

train_set, valid_set, test_set = load_mnist('.//data')
n_hidden = 100
n_visible = 784
rbm = RBMModel(visible=RBMLayer(activation='linear', units=n_visible, use_bias=True, sampled=False), hidden=RBMLayer(activation='sigmoid', units=n_hidden, use_bias=True, sampled=True))
rbm.compile(cd(1, lr=1e-2), unstack=True, kernel_regularizer=l1(0.00001))
rbm.fit(train_set, batch_size=256, nb_epoch=20, verbose=1)
valid_set_reconstruction = rbm.generate(valid_set, sampled=False, n=1)
```
harmonium/rbm_utils.py contains some simple utilities to explore RBM weights.
save_weights function saves weights as separate maps for every hidden neuron. For grayscale RBM images it should print set of learned features.
save_hidden_state function saves changes in hidden states across a set of samples. For well-trained RBM it shouldn't contain continous white or black lines.

# Installation

```shell
git clone https://github.com/aby2s/harmonium.git
cd harmonium
python setup.py install
```
# Release notes
## Version 0.0.2
* Renamed to harmonium
* Manual gradient calculation changed to tensorflow optimizers with stop_gradient applied on rbm states
* Added PCD optimizer

## Version 0.0.1
* Initial release

# Contacts

Create an issue or send me an email (aby2sz@gmail.com).


