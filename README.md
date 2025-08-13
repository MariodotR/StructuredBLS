# SBLS Class Documentation

The `BLS` class implements a structured broad learning system (SBLS) model using basis functions and enhanced nodes for tasks such as classification and regression. This model leverages various sampling techniques (e.g., Gaussian, sigmoid, and GELU) for constructing basis functions, allowing it to capture diverse data patterns.


## Quick Start Guide

```python
# Initialize BLS model
model = BLS(1E-8, 10, 10, 10, sampling= ["gaussian", "gaussian"], task="classification")
#Train
train_x, train_y = ...  # Load or prepare your training dataset
train_acc, train_time, output_weights = model.train(train_x, train_y)
print("Training Accuracy:", train_acc)
print("Training Time:", train_time)
#inference
test_x, test_y = ...  # Load or prepare your testing dataset
test_acc, test_time, predictions = model.inference(test_x, test_y)
print("Test Accuracy:", test_acc)
print("Inference Time:", test_time)
```

## Class Overview

### Methods
The class includes the following main methods:

1. **`train`**: Trains the model on the provided dataset, generating feature and enhanced nodes.
2. **`inference`**: Runs inference on a test dataset to compute accuracy or other metrics.
3. **`retrieve`**: Retrieves learned representations for specific features.
4. **`AddEnhanceNodes`**: Incrementally adds more enhanced nodes to improve the model's performance.

---

## Method Descriptions

### `__init__(self, c, d, n, m, sampling, task, autoencoder=False, epsilon=1, individual_experiment=False)`
Initializes the BLS class with specified parameters.

**Parameters:**
- `c` (float): Regularization coefficient.
- `d` (int): Dimension of each latent feature vector.
- `n` (int): Number of feature nodes.
- `m` (int): Number of enhanced nodes.
- `sampling` (str): Sampling type for basis functions (`"gaussian"`, `"sigmoid"`, or `"gelu"`).
- `task` (str): Task type (`"classification"` or `"regression"`).
- `autoencoder` (bool): Whether to apply autoencoding to feature nodes.
- `epsilon` (float): Scale factor for noise.
- `individual_experiment` (bool): If `True`, computes individual performance metrics per node groups.

### `train(self, train_x, train_y)`
Trains the BLS model by generating and scaling feature nodes and enhanced nodes.

**Parameters:**
- `train_x` (array): Training data features.
- `train_y` (array): Training data labels.

**Returns:**
- Tuple of training accuracy, training time, and output weights.

### `inference(self, test_x, test_y)`
Performs inference on test data using the trained BLS model.

**Parameters:**
- `test_x` (array): Test data features.
- `test_y` (array): Test data labels.

**Returns:**
- Tuple of test accuracy, inference time, and output layer predictions.

### `retrieve(self, digit, index, x)`
Retrieves learned curves for specific feature representations, useful for interpretability.

**Parameters:**
- `digit` (int): Target class index.
- `index` (int): Index within the output layer.
- `x` (array): Input features.

**Returns:**
- Learned curve, representation in input space, and representative examples.

### `AddEnhanceNodes(self, steps, nodes_per_step, train_x, train_y, test_x, test_y)`
Incrementally adds more enhanced nodes to the BLS model.

**Parameters:**
- `steps` (int): Number of incremental steps.
- `nodes_per_step` (int): Number of nodes added per step.
- `train_x` (array): Training data.
- `train_y` (array): Training labels.
- `test_x` (array): Test data.
- `test_y` (array): Test labels.

**Returns:**
- Training and testing accuracy and time for each step.

---



# Datasets:

Mnist: https://keras.io/api/datasets/mnist/

Coil100: https://www.kaggle.com/datasets/sotheysean/coil100/data

Isolet: https://jundongl.github.io/scikit-feature/datasets.html

Abalone: https://archive.ics.uci.edu/dataset/1/abalone

Weather Izmir: https://sci2s.ugr.es/keel/dataset.php?cod=78

Housing: https://keras.io/api/datasets/california_housing/

Arcene: https://archive.ics.uci.edu/dataset/167/arcene

GenExp: https://www.kaggle.com/datasets/crawford/gene-expression/data

P53: https://archive.ics.uci.edu/dataset/188/p53+mutants
