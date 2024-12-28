#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

from .layers import Layer
from .network_units import SupervisedUnit
from graphviz import Digraph
from warnings import warn


class CascadeNetwork(Digraph):
    def __init__(self, name="Cascade Network"):
        Digraph.__init__(
            self, comment=name, graph_attr={"splines": "ortho", "remincross": "true"}
        )
        self.layers = []
        self._target_layer = None
        self.networks = []
        self._graph_networks = {}
        self._name = name

    def add_cascade_layer(self, network_model: object, connect_to_inlayer=-1) -> object:
        # TODO Filter should only be settable for a single layer
        if 0 == len(self.layers):
            self._add_target_layer()
            self._add_layer()  # Create First Layer  #######

        if network_model.name in self._get_network_names():
            raise ValueError("Network name already exsisting in cascade")

        last_layer = self.layers[connect_to_inlayer]  # get last layer of all
        self._graph_network(network_model)
        self._connect_network(network_model, in_layer=last_layer)

        # if not (filter is None):
        #    self._add_layer(filter)
        # else:
        # self._add_layer()  # Create Second Layer  #######
        self._add_layer()
        # if filter_layer_connections is None:
        for layer in self.layers[:-1]:
            self._connect_layers(layer, self.layers[-1])
        # else:
        #    connect_to = [layer for layer in self.layers[:-1] if layer.position not in filter_layer_connections]
        #    for layer in connect_to:
        #        self._connect_layers(layer, self.layers[-1])

        if isinstance(network_model, SupervisedUnit):
            self._connect_network(network_model, target_layer=self._target_layer)
        self._connect_network(network_model, out_layer=self.layers[-1])

        self.networks.append(network_model)

    def remove_cascade_layer(self):
        self.layers.pop(-1)
        self.networks.pop(-1)

    def _get_network_names(self):
        return [net.name for net in self.networks]  # TODO inefficient. save static?

    def _graph_network(self, network_model):
        network_name = network_model.name + " " + str(len(self._graph_networks))
        self._graph_networks[network_model] = network_name
        self.node(
            network_name,
            str(network_model.__class__.__name__)
            + " "
            + str(len(self._graph_networks)),
            shape="square",
        )

    def _add_layer(self, filter=None):
        tmp_layer = Layer(len(self.layers), filter=filter)
        if filter is None:
            self.node(
                str(len(self.layers)), str(len(self.layers)), shape="invtrapezium"
            )  # create layer node
        else:
            self.node(
                str(len(self.layers)),
                str(len(self.layers)) + "(" + str(filter) + ")",
                shape="invtrapezium",
            )  # create layer node
        self.layers.append(tmp_layer)

    def _add_target_layer(self):
        self.node(
            str(sys.float_info.max), "Target Layer", shape="egg"
        )  # create layer node
        self._target_layer = Layer(sys.float_info.max)

    def _connect_layers(self, in_layer, out_layer):
        self.edge(str(in_layer.position), str(out_layer.position))  # create layer edge
        in_layer.connect_and_send_data_to(out_layer)

    def _connect_network(
        self, network_model, in_layer=None, out_layer=None, target_layer=None
    ):
        if in_layer:
            self.edge(str(in_layer.position), self._graph_networks[network_model])
            in_layer.connect_layer_as_input_for(network_model)
        if out_layer:
            self.edge(
                self._graph_networks[network_model],
                str(out_layer.position),
                label=str(list(network_model.parameters().keys())),
            )
            out_layer.connect_layer_as_output_for(network_model)
        if target_layer:
            self.edge(str(target_layer.position), self._graph_networks[network_model])
            target_layer.connect_layer_as_target_for(network_model)

    def print_state(self, state):
        for state_i in state.keys():
            print(state_i, state[state_i].shape)

    def _activate_casscade(
        self,
        Xs: object,
        Y: object = None,
        train: object = False,
        show_error: object = False,
    ) -> object:
        self.layers[0].set_state(Xs)
        loss = None
        for i, network in enumerate(self.networks):
            if train:
                self._target_layer.set_state(Y)
                if not network.initialized and Y is not None:
                    params = network.optimize_unit()
                    network.initialize_unit(params)
                loss = network.casscade_train()
                if show_error:
                    print("training error ", str(loss), " of layer: ", str(i))
            if not network.initialized and Y is None:
                warn(
                    "Applying/testing the cascade without initialized networks could lead to errors!"
                )
            network.casscade_apply()
        return loss

    def train_networks(self, Xs, Y, show_error=False):
        return self._activate_casscade(Xs, Y, True, show_error)

    def predict(self, Xs, output_dim=3):
        self.apply_data(Xs)
        last_l_out = self.get_last_layer_output()
        return last_l_out[:, -output_dim:]

    def apply_data(self, Xs):
        self._activate_casscade(
            Xs, None
        )  # TODO Refactore with observer pattern for adding new network

    def draw_graph(self, path: str = "/"):
        if path[-1] != "/":
            path += "/"
        self.render(path + self._name)

    def get_target_layer(self):
        return self._target_layer

    def get_last_layer_output(self):
        """
        returns the state of the last layer
        """
        return self.layers[-1].get_data()
