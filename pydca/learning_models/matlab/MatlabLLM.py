#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from .NetworkUnit import NetworkUnit
from pathlib import Path


class MatlabLLM(NetworkUnit):
    def __init__(self, eng, name, specs):
        super().__init__()
        cwd = os.getcwd()
        eng.eval("cd " + os.environ["MLT_ROOT"].replace("~", str(Path.home())) + "/src")
        eng.eval("mlt_include_paths")
        eng.eval("cd " + cwd)
        self.eng = eng
        self.name = name
        self.initialized = False
        self._specs = specs

    def initialize_network_model(self, X, Y):
        if not self.initialized:
            self.eng.workspace.llm_X = X
            self.eng.workspace.llm_Y = Y
            self.eng.workspace.llm_inpDim = X.shape[1]
            self.eng.workspace.llm_outDim = Y.shape[1]
            self.eng.eval("llm_specs = " + self._specs + ";")
            self.eng.eval(
                self.name + "= LLMModified(llm_inpDim, llm_outDim, llm_specs);"
            )
            self.eng.workspace.rand_idx = np.random.randint(X.shape[0])
            self.eng.eval(self.name + ".init(llm_X(rand_idx,:), llm_Y(rand_idx,:))")
            self.initialized = True

    def train(self):
        yhats = []
        X = self.eng.workspace.X
        Y = self.eng.workspace.Y
        if (
            len(Y.shape) < 2
        ):  # HACK because of Matlabs stupidity of converting single valued arrays to scalars
            Y = np.asarray([Y]).T
        error = 99999.0  # N
        last_error = 999999.0
        print("Train Matlab LLM")  # TODO use LOG
        while np.abs(error - last_error) > 0.001:
            last_error = error

            self.train_one_sample(X, Y)
            yhats = self.apply(X)
            # for i, sample in enumerate(tqdm.tqdm(X)):
            #     y = Y[i]
            #     if len(y.shape) < 2: # HACK same as before
            #         y = np.asarray([y])
            #     self.train_one_sample(sample,y)
            #     yhat = self.apply(sample)
            #     yhats.append(yhat)
            # convert list to array
            # pdb.set_trace()
            # yhats = np.array(yhats)
            error = self._calc_error_vector_length(Y, yhats)
            print(error)
        return yhats

    # def train(self):
    #    self.eng.eval(self.name+".train(X,Y')")

    def casscade_train(self):
        in_data = self.input_layer.get_data()
        out_data = self.target_layer.get_data()
        self.eng.workspace.X = in_data
        self.eng.workspace.Y = out_data
        yhats = self.train()
        return self._calc_error_vector_length(out_data, yhats)

    def train_extern(self, X, Y):
        self.eng.workspace.X = X
        self.eng.workspace.Y = Y
        self.train()

    def train_one_sample(self, x, y):
        self.eng.workspace.x = x
        self.eng.workspace.y = y
        self.eng.eval(self.name + ".train(x,y)")

    def apply(self, x):
        self.eng.workspace.x = x
        self.eng.eval("llm_single_output = " + self.name + ".apply(x)")
        return self.eng.get("llm_single_output")

    def casscade_apply(self):
        in_data = self.input_layer.get_data()
        self.eng.workspace.X = in_data
        llm_output = self.activate()
        self.set_output(llm_output)

    def activate(self):
        self.eng.eval("llm_output = " + self.name + ".apply(X)")
        return self.eng.get("llm_output")

    def activate_extern(self, X):
        self.eng.workspace.X = X
        llm_output = self.activate()
        return self.set_output(llm_output)

    def set_output(self, llm_output):
        np_a = llm_output
        if self.output_layer is not None:
            self.output_layer.set_state(np_a)
        return np_a
