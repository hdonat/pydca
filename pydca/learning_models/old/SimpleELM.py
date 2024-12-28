#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from pydca.pydca.cascade.network_units import SupervisedUnit


class ELM(SupervisedUnit):
    def __init__(self, name, hidden_size, bias):
        super().__init__()
        self._bias = bias
        self.name = name
        self.initialized = False
        self._hidden_size = hidden_size

    def initialize_network_model(self, X, Y):
        self._W_in = np.random.randn(self._hidden_size, X.shape[1])
        self._W_out = np.random.randn(Y.shape[1], self._hidden_size)
        self._current_H_state = np.zeros((1, self._hidden_size))
        self.initialized = True

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _backward(self, T):
        return T.T.dot(np.linalg.pinv(self._current_H_state)).T

    def _calc_hidden_state(self, X):
        _bias = np.ones(X.shape) * self._bias
        self._current_H_state = self._sigmoid(self._W_in.dot((X + _bias).T))

    def train(self, X, T):
        self._calc_hidden_state(X)
        self._W_out = self._backward(T)
        return self.activate(X)

    def train_extern(self, X, Y):
        self.train(X, Y)

    def _forward(self, X):
        self._calc_hidden_state(X)
        return self._W_out.T.dot(self._current_H_state).T

    def activate(self, X):
        return self._forward(X)

    def activate_extern(self, X):
        elm_output = self.activate(X)
        return self.set_output(elm_output)

    def casscade_train(self):
        in_data = self.input_layer.get_data()
        out_data = self.target_layer.get_data()
        yhats = self.train(in_data, out_data)
        return self._calc_error_vector_length(out_data, yhats)

    def casscade_apply(self):
        in_data = self.input_layer.get_data()
        elm_output = self.activate(in_data)
        self.set_output(elm_output)

    def set_output(self, elm_output):
        np_a = elm_output
        if self.output_layer is not None:
            self.output_layer.set_state(np_a)
        return np_a


class RRELM(SupervisedUnit):
    def __init__(self, loss_function, parameters={}, name: str = None):
        super().__init__(loss_function, parameters, name)
        self._hidden_size = parameters["hidden_layer_dim"]
        self._bias = parameters["bias"]
        self._regularization = parameters["alpha"] * np.identity(self._hidden_size)
        self._a = np.ones((1, self._hidden_size))
        self._b = self._bias * (
            2 * np.random.random((1, self._hidden_size))
            - np.ones((1, self._hidden_size))
        )
        self._bip = parameters["bip"]
        self.initialized = False

    def initialize_unit(self):
        X = self.input_layer.get_data()
        Y = self.target_layer.get_data()
        self._W_in = np.random.randn(self._hidden_size, X.shape[1])
        try:
            shape = (Y.shape[0], Y.shape[1])
        except IndexError:
            shape = (Y.shape[0], 1)
        self._W_out = np.random.randn(shape[1], self._hidden_size)
        self._current_H_state = np.zeros((1, self._hidden_size))
        self.initialized = True

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * self._a * x - self._b))

    def _backward(self, T):
        inverse = np.linalg.pinv(
            self._current_H_state.dot(self._current_H_state.T) + self._regularization
        )
        return inverse.dot(self._current_H_state.dot(T)).T

    def _calc_hidden_state(self, X):
        self.H_before_f = self._W_in.dot(X.T)
        self._current_H_state = self._sigmoid(self.H_before_f.T).T

    def train(self, X, T):
        if self._bip:
            self.bip_extension(X, T)
        self._calc_hidden_state(X)
        self._W_out = self._backward(T)

    def train_extern(self, X, Y):
        self.train(X, Y)

    def _forward(self, X):
        self._calc_hidden_state(X)
        return self._W_out.dot(self._current_H_state)

    def activate(self, X):
        return self._forward(X).T

    def activate_extern(self, X):
        elm_output = self.activate(X)
        return self.set_output(elm_output)

    def bip_extension(self, X, T):
        input_activation = self._W_in.dot(X.T)
        for i in range(T.shape[0]):
            inverse_test = np.dot(np.linalg.pinv(self._W_out), T[i])
            finv = -1 * np.log(1.0 / inverse_test - 1)
            # clean numerical erros
            where_are_NaNs = np.isnan(finv)
            finv[where_are_NaNs] = 0
            where_are_inf = np.isinf(finv)
            finv[where_are_inf] = 99
            si = np.asarray([input_activation[:, i]])
            si = np.sort(si)
            t = np.sort(finv)
            phi = np.vstack([si, np.ones((1, si.shape[1]))])
            vi = np.linalg.pinv(phi.T).dot(t.T)
            self._a[0, i] = vi[0]
            self._b[0, i] = vi[1]

    def casscade_train(self):
        in_data = self.input_layer.get_data()
        out_data = self.target_layer.get_data()
        self.train(in_data, out_data)
        yhats = self.activate(in_data)
        return self._calc_loss_function(out_data, yhats)

    def casscade_apply(self):
        in_data = self.input_layer.get_data()
        elm_output = self.activate(in_data)
        self.set_output(elm_output)

    def set_output(self, elm_output):
        np_a = elm_output
        if self.output_layer is not None:
            self.output_layer.set_state(np_a)  # .reshape(-1,1))
        return np_a

    def parameters(self):
        return {"hidden_layer_dim": None}


class ReLUELM(ELM):
    def __init__(
        self,
        name,
        hidden_size,
        bias,
    ):
        super().__init__(name, hidden_size, bias)

    def _sigmoid(self, x):
        return self.ReLU(x)

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1.0 * (x > 0)


class RRReLUELM(RRELM):
    def __init__(self, name, hidden_size, bias, alpha, bip=False):
        super().__init__(name, hidden_size, bias, alpha, bip=False)

    def _sigmoid(self, x):
        return self.ReLU(x)

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1.0 * (x > 0)


class LeakyReLUELM(ELM):
    def __init__(self, name, hidden_size, bias):
        super().__init__(name, hidden_size, bias)

    def _sigmoid(self, x):
        return self.ReLU(x)

    def ReLU(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def dlrelu(dl, x, alpha=0.01):
        """dl and x have same shape."""
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx * dl


class RRLeakyReLUELM(RRELM):
    def __init__(self, name, hidden_size, bias, alpha, bip=False):
        super().__init__(name, hidden_size, bias, alpha, bip=False)

    def _sigmoid(self, x):
        return self.ReLU(x)

    def ReLU(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def dlrelu(dl, x, alpha=0.01):
        """dl and x have same shape."""
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx * dl
