# NN Framework

This script is an implementation of a primitive machine learning framework in Python. The goal is to provide a basic
understanding of how machine learning frameworks work by implementing key components such as activation functions,
layers, losses, and a simple neural network.



## Usage


**Create a network**:To create a network build a list of layers and then initialize Sequential class with the list. 
```python
module_list = [
        Linear(input_size=2, output_size=50),
        Tanh(),
        Linear(input_size=50, output_size=30),
        Tanh(),
        Linear(input_size=30, output_size=10),  # Add or modify layers as needed
        Sigmoid()
    ]
    custom_network = Sequential(module_list)
```

**Train**:

```python
new_X, new_T = load_new_dataset()  # Load or prepare your new dataset
new_network = create_custom_network()
new_loss = MSE()

for epoch in range(num_epochs):
    error = train_one_step(new_network, new_loss, learning_rate, new_X, new_T)

```

**Prediction**:
```python
new_data_for_prediction = load_new_data_for_prediction()  # Load or prepare your new dataset for prediction
prediction = new_network.forward(new_data_for_prediction)
```

### Example

The script uses a simple two-spiral dataset for training, and the final boundary plot should show how well the trained
model separates the two classes. The vidoe shows how the model improves after epochs.

[![Sample Video](https://img.youtube.com/vi/_izhkazRGqE/0.jpg)](https://www.youtube.com/watch?v=_izhkazRGqE)


## Components

Here's important components of the code:

1. **Activation Functions: Tanh and Sigmoid**
    - `Tanh` class implements the hyperbolic tangent activation function.
    - `Sigmoid` class implements the sigmoid activation function.
    - Both classes have `forward` and `backward` methods for the forward and backward passes.

2. **Linear Layer:**
    - `Linear` class represents a linear layer (fully connected layer) in a neural network.
    - It has `forward` and `backward` methods for the forward and backward passes.
    - The weights (`W`) and bias (`b`) are initialized during object creation.
    - The `forward` method calculates the output by performing a linear transformation on the input.
    - The `backward` method calculates the gradients with respect to the weights, bias, and input.

3. **Sequential Layer:**
    - `Sequential` class represents a sequential composition of layers.
    - It takes a list of modules (layers) during initialization.
    - It has `forward` and `backward` methods that sequentially call the forward and backward methods of each module.

4. **Mean Squared Error (MSE) Loss:**
    - `MSE` class implements the mean squared error loss.
    - It has `forward` and `backward` methods for calculating the loss and its gradient.

5. **Training Functions:**
    - `update_module` function updates the weights and biases of a module based on gradients and learning rate.
    - `train_one_step` function performs one step of training by forward and backward passes, calculating the error, and
      updating the weights.

6. **Neural Network Creation:**
    - `create_network` function creates a specific neural network structure using the defined layers.

7. **Gradient Checking:**
    - `gradient_check` function checks the correctness of the backward pass by comparing analytical and numerical
      gradients.

8. **Main Function:**
    - `main` function performs gradient checking and then trains the neural network on a two-spiral dataset.

9. **Data Generation:**
    - `twospirals` function generates a two-spiral dataset.

10. **Visualization:**
    - The script uses `matplotlib` for data and boundary visualization during training.


### Run

To run this script, you need to have `numpy` and `matplotlib` installed (`pip3 install numpy matplotlib`).
