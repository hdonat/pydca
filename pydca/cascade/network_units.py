#!/usr/bin/python
#-*- coding: utf-8 -*-

'''
This module contains the
'''

from abc import ABC, abstractmethod

class CascadeUnit(ABC):
    """Abstract unit for leaner units in the network
    """
    def __init__(self, name = None, parameters ={}):
        self.network_model = None
        self.input_layer = None
        self.output_layer = None
        self.target_layer = None
        self.initialized = False
        self._parameters = parameters
        if name is None:
            self.name = str(id(self))
        else:
            self.name = name

    @abstractmethod
    def initialize_unit(self, ):
        """Abstract method
        """
        pass

    @abstractmethod
    def train(self, input_x, output_y):
        """Abstract method
        """
        pass

    @abstractmethod
    def activate(self, ):
        """Abstract method
        """
        pass

    @abstractmethod
    def set_output(self, ):
        """Abstract method
        """
        pass

    @abstractmethod
    def casscade_train(self):
        pass

    def casscade_apply(self):
        in_data = self.input_layer.get_data()
        output = self.activate(in_data)
        self.set_output(output)

    @abstractmethod
    def parameters(self):
        pass




class SupervisedUnit(CascadeUnit):
    """RegressorUnit is a normal cascade unit but with a loss metric e.g., for supervised learning.
    :param loss_function: externally defined loss function
    :type loss_function: python function
    """
    def __init__(self, loss_function, parameters, name: str = None):
        """Constructor method
        """
        super().__init__(name, parameters)
        self._loss_function = loss_function

    def _calc_loss_function(self, target, prediction):
        """Calls injected loss function
        """
        return self._loss_function(target, prediction)

    @abstractmethod
    def initialize_unit(self, input_x, input_y):
        """Abstract method
        """
        pass

    @abstractmethod
    def optimize_unit(self):
        """Abstract method
        """
        pass


class UnsupervisedUnit(CascadeUnit):
    """RegressorUnit is a normal cascade unit but with a loss metric e.g., for supervised learning.
    :param loss_function: externally defined loss function
    :type loss_function: python function
    """
    def __init__(self, loss_function, parameters, name: str = None):
        """Constructor method
        """
        super().__init__(name, parameters)
        self._loss_function = loss_function


    def _calc_loss_function(self, target, prediction):
        """Calls injected loss function
        """
        return self._loss_function(target, prediction)

    @abstractmethod
    def initialize_unit(self, input_x):
        """Abstract method
        """
        pass


class DataUnit(CascadeUnit):

    def __init__(self, data, parameters, name: str = None):
        """Constructor method
        """
        super().__init__(name, parameters)
        self._data = data

    def initialize_unit(self):
        pass

    def train(self, input_x, output_y):
        pass

    def activate(self):
        pass

    def set_output(self):
        pass

    def casscade_train(self):
        pass

    def parameters(self):
        pass

    def casscade_apply(self):
        self.output_layer.set_state(self._data)
