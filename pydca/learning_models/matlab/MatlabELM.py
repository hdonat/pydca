import os

from pathlib import Path

from .NetworkUnit import NetworkUnit


class MatlabELM(NetworkUnit):
    def __init__(self, eng, name):
        super().__init__()
        # try:
        #     self.eng = matlab.engine.start_matlab()
        # except EngineError:
        #     print("EngineError: Check if License Server is available.")
        #     return
        # print("Include MLT Toolbox")
        # self.eng.include_paths(nargout=0)
        cwd = os.getcwd()
        eng.eval("cd " + os.environ["MLT_ROOT"].replace("~", str(Path.home())) + "/src")
        eng.eval("mlt_include_paths")
        eng.eval("cd " + cwd)
        self.eng = eng
        self.name = name
        self.initialized = False

    def initialize_network_model(
        self,
    ):
        pass

    def train(self):
        # pdb.set_trace()
        self.eng.eval(self.name + ".train(X,Y)")
        return self.activate()

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

    def activate(self):
        self.eng.eval("simpleELMOutput = " + self.name + ".apply(X)")
        elm_output = self.eng.get("simpleELMOutput")
        return elm_output

    def casscade_apply(self):
        in_data = self.input_layer.get_data()
        self.eng.workspace.X = in_data
        elm_output = self.activate()
        self.set_output(elm_output)

    def activate_extern(self, X):
        self.eng.workspace.X = X
        elm_output = self.activate()
        return self.set_output(elm_output)

    def set_output(self, elm_output):
        np_a = elm_output
        # np_a = np_a.reshape(elm_output.size).transpose()
        # pdb.set_trace()
        if self.output_layer is not None:
            self.output_layer.set_state(np_a)
        return np_a


class SimpleELM(MatlabELM):
    def __init__(
        self, eng, name, hidden_size, reg=0.0000006, normalize=False, bip=False
    ):
        MatlabELM.__init__(self, eng, name)
        self._hsize = hidden_size
        self._bip = bip
        self._normalize = normalize
        self._reg = reg

    def initialize_network_model(self, X, Y):
        self.eng.workspace.X = X
        self.eng.workspace.Y = Y

        bip = 1 if self._bip else 0
        self.eng.eval(
            self.name
            + "= ELM(size(X,2), size(Y,2), {'class', 'ELM'; 'hidDim',"
            + str(self._hsize)
            + "; 'BIP', "
            + str(bip)
            + "})"
        )
        if not self._normalize:
            self.eng.eval(self.name + ".inpNormalization = 0")
            self.eng.eval(self.name + ".outNormalization = 0")
        self.eng.eval(self.name + ".reg =" + str(self._reg))
        self.eng.eval(self.name + ".init(X,Y)")
        self.initialized = True
