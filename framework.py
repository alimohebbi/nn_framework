#!/usr/bin/env python3

import numpy as np
from scipy.special import expit


class Tanh:
    def __init__(self):
        self.saved_variables = None

    def forward(self, x):
        # Implement
        result = np.tanh(x)
        # End

        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, error):
        tanh_x = self.saved_variables["result"]

        # Implement
        d_x = 1 - np.power(tanh_x, 2)
        d_x = error * d_x
        assert d_x.shape == tanh_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, tanh_x.shape)

        self.saved_variables = None
        return None, d_x


class Sigmoid:
    def __init__(self):
        self.saved_variables = None

    def forward(self, x):
        # Implement

        result = expit(x)

        # End

        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, error):
        sigmoid_x = self.saved_variables["result"]

        # Implement

        d_x = sigmoid_x * (1 - sigmoid_x)
        d_x = error * d_x
        # End
        assert d_x.shape == sigmoid_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, sigmoid_x.shape)

        self.saved_variables = None
        return None, d_x


class Linear:
    def __init__(self, input_size, output_size):
        self.var = {
            "W": np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (input_size, output_size)),
            "b": np.zeros(output_size, dtype=np.float32)
        }
        self.saved_variables = None

    def forward(self, inputs):
        x = inputs
        W = self.var['W']
        b = self.var['b']

        # Implement
        # Save your variables needed in backward pass to self.saved_variables.

        y = x.dot(W) + b
        self.saved_variables = {'x': x}

        # End
        return y

    def backward(self, error):
        # Implement
        x = self.saved_variables['x']
        dW = x.T.dot(error)
        b_column = np.ones((1, error.shape[0]))
        db = b_column.dot(error)[0]
        w = self.var['W']
        d_inputs = w.T
        d_inputs = error.dot(d_inputs)
        # End

        assert d_inputs.shape == x.shape, "Input: grad shape differs: %s %s" % (d_inputs.shape, x.shape)
        assert dW.shape == self.var["W"].shape, "W: grad shape differs: %s %s" % (dW.shape, self.var["W"].shape)
        assert db.shape == self.var["b"].shape, "b: grad shape differs: %s %s" % (db.shape, self.var["b"].shape)

        self.saved_variables = None
        updates = {"W": dW,
                   "b": db}
        return updates, d_inputs


class Sequential:
    def __init__(self, list_of_modules):
        self.modules = list_of_modules

    class RefDict(dict):
        def add(self, k, d, key):
            super().__setitem__(k, (d, key))

        def __setitem__(self, k, v):
            assert k in self, "Trying to set a non-existing variable %s" % k
            ref = super().__getitem__(k)
            ref[0][ref[1]] = v

        def __getitem__(self, k):
            ref = super().__getitem__(k)
            return ref[0][ref[1]]

        def items(self):
            for k in self.keys():
                yield k, self[k]

    @property
    def var(self):
        res = Sequential.RefDict()
        for i, m in enumerate(self.modules):
            if not hasattr(m, 'var'):
                continue

            for k in m.var.keys():
                res.add("mod_%d.%s" % (i, k), m.var, k)
        return res

    def update_variable_grads(self, all_grads, module_index, child_grads):
        if child_grads is None:
            return all_grads

        if all_grads is None:
            all_grads = {}

        for name, value in child_grads.items():
            all_grads["mod_%d.%s" % (module_index, name)] = value

        return all_grads

    def forward(self, input):
        # Implement
        output = input
        for module_index in range(len(self.modules)):
            module = self.modules[module_index]
            output = module.forward(output)
        return output

        # End

    def backward(self, error):
        variable_grads = None

        for module_index in reversed(range(len(self.modules))):
            module = self.modules[module_index]

            # Implement

            module_variable_grad, module_input_grad = module.backward(error)

            # End
            error = module_input_grad
            variable_grads = self.update_variable_grads(variable_grads, module_index, module_variable_grad)

        return variable_grads, module_input_grad


class MSE:
    def __init__(self):
        self.saved_variables = None

    def forward(self, prediction, target):
        Y = prediction
        T = target
        n = prediction.size

        # Implement
        # Don't forget to save your variables needed for backward to self.saved_variables..
        self.saved_variables = {'Y': Y, 'T': T}

        meanError = np.transpose(Y - T).dot(Y - T) / (2 * n)

        # End
        return meanError 

    def backward(self):
        # Implement
        y = self.saved_variables['Y']
        t = self.saved_variables['T']
        d_prediction = (y - t) / y.size

        # End
        assert d_prediction.shape == y.shape, "Error shape doesn't match prediction: %d %d" % \
                                              (d_prediction.shape, y.shape)

        self.saved_variables = None
        return d_prediction


def update_module(network, learning_rate, var_grads):
    for i in range(len(network.modules)):
        module = network.modules[i]
        if not hasattr(module, 'var'):
            continue
        old_w = module.var['W']
        old_b = module.var['b']
        dw = var_grads["mod_%d.W" % i]
        db = var_grads["mod_%d.b" % i]
        new_w = old_w - (learning_rate * dw)
        new_b = old_b - (learning_rate * db)
        module.var['W'] = new_w
        module.var['b'] = new_b


def train_one_step(model, loss, learning_rate, inputs, targets):
    # Implement
    predict = model.forward(inputs)
    error = loss.forward(predict, targets)
    loss_back = loss.backward()
    var_grads, _ = model.backward(loss_back)
    update_module(model, learning_rate, var_grads)

    # End
    return error


def create_network():
    # Implement
    module_list = Linear(2, 50), Tanh(), Linear(50, 30), Tanh(), Linear(30, 1), Sigmoid()
    network = Sequential(module_list)

    # End
    return network


def gradient_check():
    X, T = twospirals(n_points=10)
    NN = create_network()
    eps = 0.0001

    loss = MSE()
    loss.forward(NN.forward(X), T)
    variable_gradients, _ = NN.backward(loss.backward())

    all_succeeded = True

    # Check all variables. Variables will be flattened (reshape(-1)), in order to be able to generate a single index.
    for key, value in NN.var.items():
        variable = NN.var[key].reshape(-1)
        variable_gradient = variable_gradients[key].reshape(-1)
        success = True

        if NN.var[key].shape != variable_gradients[key].shape:
            print("[FAIL]: %s: Shape differs: %s %s" % (key, NN.var[key].shape, variable_gradients[key].shape))
            success = False
            break

        # Check all elements in the variable
        for index in range(variable.shape[0]):
            var_backup = variable[index]

            # Implement
            grad_backup = variable_gradient[index]

            variable[index] = var_backup + eps
            y_plus = loss.forward(NN.forward(X), T)

            variable[index] = var_backup - eps
            y_minus = loss.forward(NN.forward(X), T)
            analytic_grad = grad_backup
            numeric_grad = (y_plus - y_minus) / (2 * eps)

            # End

            variable[index] = var_backup
            if abs(numeric_grad - analytic_grad) > 0.00001:
                print("[FAIL]: %s: Grad differs: numerical: %f, analytical %f" % (key, numeric_grad, analytic_grad))
                success = False
                break

        if success:
            print("[OK]: %s" % key)

        all_succeeded = all_succeeded and success

    return all_succeeded


########################################################
# Nothing to do past this line.
########################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0xDEADBEEF)

    plt.ion()


    def twospirals(n_points=120, noise=1.6, twist=420):
        """
         Returns a two spirals dataset.
        """
        np.random.seed(0)
        n = np.sqrt(np.random.rand(n_points, 1)) * twist * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        X, T = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points), np.ones(n_points))))
        T = np.reshape(T, (T.shape[0], 1))
        return X, T


    fig, ax = plt.subplots()


    def plot_data(X, T):
        ax.scatter(X[:, 0], X[:, 1], s=40, c=T.squeeze(), cmap=plt.cm.Spectral)


    def plot_boundary(model, X, targets, threshold=0.0):
        ax.clear()
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        y = model.forward(X_grid)
        ax.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
        plot_data(X, targets)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([x_min, x_max])
        plt.show()
        plt.draw()
        plt.pause(0.001)


    def main():
        print("Checking the network")
        if not gradient_check():
            print("Failed. Not training, because your gradients are not good.")
            return
        print("Done. Training...")

        X, T = twospirals(n_points=200, noise=1.6, twist=600)
        NN = create_network()
        loss = MSE()

        learning_rate = 0.1

        for i in range(20000):
            curr_error = train_one_step(NN, loss, learning_rate, X, T)
            if i % 200 == 0:
                print("step: ", i, " cost: ", curr_error)
                plot_boundary(NN, X, T, 0.5)

        plot_boundary(NN, X, T, 0.5)
        print("Done. Close window to quit.")
        plt.ioff()
        plt.show()


    main()
