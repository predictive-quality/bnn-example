"""
Copyright 2020 Simon Cramer, Meike Huber, Robert H. Schmitt - RWTH AACHEN UNIVERSITY

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from absl import app, logging

import bnn
import evaluation


def main(argv):
    #Settings
    n_test = 60 # Total amount of samples to use for evaluation
    batch_size = 4 # Batch size for BNN training
    epochs = 1250 # Amount of epochs to train BNN
    learning_rate = 0.001 # Learning rate of the Adam optimizer
    n_samples_eval = 5000 # How often to sample the model for each test sample
    amount_of_samples = 15 # Amount of test samples for the boxplot

    # Load data and drop index column
    X = np.genfromtxt('x.csv',delimiter=',',skip_header=1,dtype=np.float)[:,1:]
    y = np.genfromtxt('y.csv',delimiter=',',skip_header=1,dtype=np.float)[:,1:]
    
    # Split data randomly into train and test set
    X_train, y_train, X_test, y_test = bnn.prepare_dataset(X,y,n_test)

    # Train the model
    model = bnn.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)

    # Run evaluation
    y_hats_mean = evaluation.evaluate(model, X_test, y_test, n_samples_eval=n_samples_eval)
    evaluation.boxplot_predictions(y_hats_mean, y_test, amount_of_samples=amount_of_samples)
    evaluation.plot_loss(model)

if __name__ == '__main__':
    app.run(main)
