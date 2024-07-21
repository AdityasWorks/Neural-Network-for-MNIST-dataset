# Neural Network from Scratch

This project implements a simple neural network from scratch using only NumPy and Pandas. The neural network is trained on the MNIST dataset for digit classification.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)

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

```python
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
```

### Forward Propagation

The network performs forward propagation using ReLU activation for the hidden layer and softmax for the output layer:

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```
### Backward Propagation

The gradients are calculated using backward propagation, and the parameters are updated accordingly:

```python
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
```
### Training

The network is trained using gradient descent:

```python
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
```
## Results
During training, the accuracy of the model improves progressively. By the 490th iteration, the network achieves an accuracy of approximately 85.77% on the training data:
```bash
Iteration:  490
[0 2 9 ... 9 3 0] [0 2 9 ... 4 3 5]
0.8576585365853658
```
The final weights and biases can be used for further predictions.
