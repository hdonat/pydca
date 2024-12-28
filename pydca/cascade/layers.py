#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from .filters import Filter


class LayerState(dict):
    """
    State representation for layer states
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]


class Layer:
    """
    Basic Layer Class
    """

    def __init__(self, number, label: str = None, filter=None, restrict_output=None):
        self.current_state = {}
        self.connected_layers = []
        self._position = number
        self._label = label

        if not filter:
            self._filter = Filter()
        else:
            self._filter = filter

        self._restrict_output = restrict_output

    def connect_layer_as_input_for(self, network):
        network.input_layer = self

    def connect_layer_as_output_for(self, network):
        network.output_layer = self

    def connect_layer_as_target_for(self, network):
        network.target_layer = self

    @property
    def position(self):
        if self._label is None:
            return self._position
        else:
            return self._label

    @position.setter
    def position(self, position):
        self._position = position

    def connect_and_send_data_to(self, layer):
        self.connected_layers.append(layer)
        if (
            self._position in self.current_state.keys()
        ):  # TODO refactor. Do not use keys()
            last_data = self.current_state[self._position]
            self.update_data(
                self._position, self._filter.in_filter(last_data, self._position)
            )

    def set_state(self, data):
        self.update_data(self._position, self._filter.in_filter(data, self._position))

    def del_state(self):
        self.current_state = {}

    def update_data(self, entity, data):
        self.current_state[entity] = (
            data  # TODO inefficent, but ! if jax or numpy used generality is lost (vec -> mat)
        )
        for layer in self.connected_layers:
            layer.update_data(entity, data)

    def get_data(self):
        if self._restrict_output is None:
            states = [
                self.current_state[key] for key in sorted(self.current_state.keys())
            ]  # get order form 0 to n
            print(self._position)
            if 1 == len(states):
                return_state = states[0]
            else:
                return_state = np.hstack(states)

            return return_state
        else:
            return self.current_state[self._restrict_output]

    def get_normalized_data(self):
        states = [
            self.current_state[key] for key in sorted(self.current_state.keys())
        ]  # get order form 0 to n
        if 1 == len(states):
            return_state = states[0]
        else:
            states = [
                state / state.shape[1] if len(state.shape) > 1 else state
                for state in states
            ]
            return_state = np.hstack(states)

        return return_state
