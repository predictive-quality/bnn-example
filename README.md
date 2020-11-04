# Uncertainty Quantification based on Bayesian neural networks for Predictive Quality
This is the code and the data set necessary to reproduce our results from the chapter _Uncertainty Quantification based on Bayesian neural networks for Predictive Quality_ in the book _Artificial Intelligence, Big Data, Data Science and Machine Learning in Statistics_ by Prof. Ansgar Steland and Kwok Leung Tsui.

Authors: Simon Cramer, Meike Huber, Robert H. Schmitt

Please cite this code or the data set as:

```
Bibtex entry is provided after succesfull publication.
```

## Installation

Clone the repository and install all requirements using `pip install -r requirements.txt` using Python>3.8.

We can report it working with:
- python==3.8.5
- matplotlib==3.3.2
- tensorflow==2.3.1
- numpy==1.18.5
- tensorflow-probability==0.11.1
- absl-py==0.11.0

## Usage

To recreate our results simply run `python main.py`. Please not that the train/test split of the data set is random and you will not excatly reproduce our figures.

You are invited to try other parameters and can change them in the file `main.py`.

## Data Set

The data set was recorded with the help of the Festo Polymer GmbH. The features (`x.csv`) are either parameters explicitly set on the injection molding machine or recorded sensor values. The target value (`y.csv`) is a crucial length measured on the parts. We measured with a high precision coordinate-measuring machine at the Laboratory for Machine Tools (WZL) at RWTH Aachen University.

If you use this data set, the citation of our publication is required! (Details see above)

## License
Copyright 2020 Simon Cramer, Meike Huber, Robert H. Schmitt - RWTH AACHEN UNIVERSITY

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
