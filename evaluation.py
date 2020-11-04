"""
Copyright 2020 Simon Cramer, Meike Huber, Robert H. Schmitt - RWTH AACHEN UNIVERSITY

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import logging


def evaluate(model, X_test, y_test, n_samples_eval = 5000):
    """Sample the model for a given test set X. Reports mean/variance of the mean predictions. Also computes MeanAbsoluteError in comparison to true target value.

    Args:
        model (tf.keras.Sequential): Fitted BNN model
        X_test (np.array): Test data set X
        y_test (np.array): True target values y
        n_samples_eval (int, optional): Amount of evaluations per sample to be computed. Defaults to 5000.
    """

    # Sample the model
    y_hats =[model(X_test) for _ in range(n_samples_eval)]

    # Get Means and repackage for error calculation:
    y_hats_mean = [y_hat.mean().numpy().flatten() for y_hat in y_hats]

    mean_pred = np.mean(y_hats_mean, axis=0)
    var_pred = np.var(y_hats_mean, axis=0)

    # Evaluation Summary
    mae = np.mean(np.abs(mean_pred-y_test))
    np.array2string(mean_pred, precision=2, separator=',',suppress_small=True)
    logging.info('Mean Prediction')
    logging.info(np.array2string(mean_pred.reshape(-1,), precision=2, separator=',',suppress_small=True))
    logging.info('Original Data')
    np.array2string(y_test, precision=2, separator=',',suppress_small=True)
    logging.info(np.array2string(y_test.reshape(-1,), precision=2, separator=',',suppress_small=True))
    logging.info('Var Prediction')
    logging.info(np.array2string(var_pred, precision=2, separator=',',suppress_small=True))
    logging.info('MAE')
    logging.info(str(mae))

    return y_hats_mean

def boxplot_predictions(y_hats_mean, y_test, amount_of_samples = 15):
    """Create a boxplot of k samples from the test data set.

    Args:
        y_hats_mean (np.array): Means from model evaluation.
        y_test (np.array): True target values y.
        amount_of_samples (int, optional): Limits the amount of samples to plot. Defaults to 15.
    """
    assert(amount_of_samples<y_test.shape[0])

    y_hats_mean = np.vstack(y_hats_mean)[:,:amount_of_samples]
    y_test = y_test[:amount_of_samples]

    _, ax1 = plt.subplots()
    ax1.boxplot(y_hats_mean, showfliers=False)

    for i, val in enumerate(y_test):
        plt.scatter(i+1,val,c='blue', marker='x')
    plt.xlabel('sample')
    plt.ylabel('quality characteristic $y_1$')
    plt.show()

def plot_loss(model):
    """Plots the loss of a given model over all training epochs.

    Args:
         model (tf.keras.Sequential): Fitted BNN model
    """
    train_loss = model.history.history['loss']
    
    plt.plot(train_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
