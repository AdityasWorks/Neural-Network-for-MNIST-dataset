# Neural Network from Scratch

This project implements a simple neural network from scratch using only NumPy and Pandas. The neural network is trained on the MNIST dataset for digit classification.

## Introduction

This repository contains a Python script that builds and trains a simple neural network on the MNIST dataset. The network uses a single hidden layer with ReLU activation and a softmax output layer.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- NumPy
- Pandas
- Matplotlib (for plotting results)

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib
```
## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch
```
2. Place the MNIST dataset CSV file (`train.csv`) in the project directory.

3. Run the script:

```bash
python neural_network.py
```
## Implementation Details

### Data Preparation

The dataset is loaded using Pandas and then converted to a NumPy array. The data is shuffled and split into training and development sets. Each pixel value is normalized by dividing by 255.

### Neural Network Initialization

The weights and biases for the two layers are initialized randomly:

### Forward Propagation

The network performs forward propagation using ReLU activation for the hidden layer and softmax for the output layer:

### Backward Propagation

The gradients are calculated using backward propagation, and the parameters are updated accordingly:

### Training

The network is trained using gradient descent:

## Results
During training, the accuracy of the model improves progressively. By the 490th iteration, the network achieves an accuracy of approximately 85.77% on the training data:
```bash
Iteration:  490
[0 2 9 ... 9 3 0] [0 2 9 ... 4 3 5]
0.8576585365853658
```
The final weights and biases can be used for further predictions.
