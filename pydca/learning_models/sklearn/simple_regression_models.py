import pdb

from sklearn import linear_model
from pydca.cascade.network_units import SupervisedUnit


class LinearModel(SupervisedUnit):
    def __init__(self, loss_function, parameters = {}, name: str = None):
        super().__init__(loss_function, parameters, name)
        self.network_model = linear_model.LinearRegression()
        self.input_layer = None
        self.output_layer = None

    def initialize_unit(self):

        self.initialized = True

    def train(self, X, Y):
        self.network_model.fit(X,Y)

    def train_extern(self, X, Y):
        self.train(X,Y)

    def activate(self, X):
        return self.network_model.predict(X)

    def activate_from_layer(self):
        in_data = self.input_layer.get_data()
        output = self.activate(in_data)
        self.set_output(output)

    def activate_extern(self,X):
        output = self.activate(X)
        return self.set_output(output)

    def set_output(self, data):
        if not (self.output_layer is None):
            self.output_layer.set_state(data)
        return data

    def casscade_train(self):
        x = self.input_layer.get_data()
        y = self.target_layer.get_data()
        self.train(x,y)

    def parameters(self):
        return {"None":None}
