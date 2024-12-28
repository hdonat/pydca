import numpy as np
from sklearn.neural_network import MLPRegressor
from pydca.cascade.network_units import SupervisedUnit
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args


class MLPModel(SupervisedUnit):
    def __init__(self, loss_function, parameters={}, name: str = None):
        super(MLPModel, self).__init__(loss_function, parameters, name)
        self.network_model = None
        self.input_layer = None
        self.output_layer = None
        self._hidden_layer_size = parameters["hidden_layer_dim"]
        self.initialized = False

    def initialize_unit(self, params):
        # if isinstance(self._hidden_layer_size,tuple):
        #    self.network_model = MLPRegressor(hidden_layer_sizes=self._hidden_layer_size, activation="tanh"
        #                                      , max_iter=10000, solver="adam")
        # else:
        #    self.network_model = MLPRegressor(hidden_layer_sizes=(self._hidden_layer_size,),activation="tanh", max_iter=10000, solver="adam")
        self.network_model = MLPRegressor(**params)
        self.initialized = True

    def train(self, X, Y):
        self.network_model.fit(X, Y)
        # plt.scatter(X,Y)
        # plt.show()
        return self.network_model.predict(X)

    def train_extern(self, X, Y):
        self.train(X, Y)

    def activate(self, X):
        return self.network_model.predict(X)

    def activate_from_layer(self):
        in_data = self.input_layer.get_data()
        output = self.activate(in_data)
        self.set_output(output)

    def activate_extern(self, X):
        output = self.activate(X)
        return self.set_output(output)

    def set_output(self, data):
        if len(data.shape) == 1:
            np_a = data.reshape(-1, 1)
        else:
            np_a = data
        if self.output_layer is not None:
            self.output_layer.set_state(np_a)
        return np_a

    def casscade_train(self):
        in_data = self.input_layer.get_data()
        out_data = self.target_layer.get_data()
        yhats = self.train(in_data, out_data)
        return self._calc_loss_function(out_data, yhats)

    def casscade_apply(self):
        in_data = self.input_layer.get_data()
        elm_output = self.activate(in_data)
        # pdb.set_trace()
        # plt.scatter(in_data,elm_output)
        # plt.show()
        self.set_output(elm_output)

    def optimize_unit(self):
        X_train = self.input_layer.get_data()
        y_train = self.target_layer.get_data()
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()
        # Define the space of hyperparameters to search
        space = [
            Integer(50, 500, name="hidden_layer_sizes"),
            Categorical(["relu", "tanh"], name="activation"),
            Real(1e-4, 1e-1, "log-uniform", name="learning_rate_init"),
            # Add other hyperparameters here
        ]

        # Define the objective function
        @use_named_args(space)
        def objective(**params):
            mlp = MLPRegressor(max_iter=10000, **params)
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_train)
            return np.mean(np.square(y_train - y_pred))

        result = gp_minimize(objective, space, n_calls=50, random_state=42)
        print(f"Best parameters: {result.x}")
        print(f"Best Mean Squared Error: {result.fun}")

        # Plot convergence of the search
        # plot_convergence(result)
        # plt.show()
        params = dict(
            zip(["hidden_layer_sizes", "activation", "learning_rate_init"], result.x)
        )
        print(f"This parameters will be used {params}")
        return params

    def parameters(self):
        return {"hidden_layer_dim": None}
