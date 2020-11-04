"""
Copyright 2020 Simon Cramer, Meike Huber, Robert H. Schmitt - RWTH AACHEN UNIVERSITY

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from absl import logging

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')


def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    """Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
    See https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression for example.

    Args:
        kernel_size (int): Size of the kernel matrix
        bias_size (int, optional): Size of the bias vector. Defaults to 0.
        dtype (dtype, optional): Datatype to avoid infering wrong formats. Defaults to None.

    Returns:
        tf.keras.Sequential: Callable
    """
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def _prior_trainable(kernel_size, bias_size=0, dtype=None):
    """Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    See https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression for example.

    Args:
        kernel_size (int): Size of the kernel matrix
        bias_size (int, optional): Size of the bias vector. Defaults to 0.
        dtype (dtype, optional): Datatype to avoid infering wrong formats. Defaults to None.

    Returns:
        tf.keras.Sequential: Callable
    """
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def prepare_dataset(X,y,n_test):
    """Splits a data set X,y randombly into training and test set.

    Args:
        X (np.array): Numpy array with n rows and k features
        y (np.array): Numpy array with n rows and t features
        n_test (int): Amount of samples for the test data set

    Returns:
        np.array: Training set x
        np.array: Training set y
        np.array: Test set x
        np.array: Test set y
    """
    assert(X.shape[0]>n_test)
    assert(y.shape[0]>n_test)
    assert(X.shape[0]==y.shape[0])

    test_index = np.random.choice(range(X.shape[0]), n_test, replace=False)
    X_test = X[test_index,:]
    y_test = y[test_index,:]
    X = np.delete(X, test_index, axis = 0)
    y = np.delete(y, test_index, axis = 0)

    return X, y, X_test, y_test

def fit( X, y, batch_size = 4, validation_split = 0.05, epochs = 1250, learning_rate=0.001):
    """Fit a BNN with mean field posterior and trainable prior to a data set X,y.

    Args:
        X (np.array): Input data x (features)
        y (np.array): Input data y (target)
        batch_size (int, optional): Batch size for BNN training. Defaults to 4.
        validation_split (float, optional): Train-validation-split to monitor training in tensorboard. Defaults to 0.05.
        epochs (int, optional): Amount of epochs to train BNN. Defaults to 1250.
        learning_rate (float, optional): Learning rate of the Adam optimizer. Defaults to 0.001.

    Returns:
        tf.keras.sequential: Fitted model
    """

    # Loss Function
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    
    kl_weight = batch_size/(X.shape[0]*(1-validation_split))

    # Build Model
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(4, _posterior_mean_field, _prior_trainable, kl_weight=kl_weight, activation=tf.nn.leaky_relu),
        tfp.layers.DenseVariational(1+1, _posterior_mean_field, _prior_trainable, kl_weight=kl_weight, activation=tf.nn.leaky_relu),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
    ])

    # Compile and train
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
    model.fit(X, y, shuffle=True, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir='tensorboard_log')])

    model.summary()

    return model
   
