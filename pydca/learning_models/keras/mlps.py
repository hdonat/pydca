from keras import layers, models
from pydca.pydca.cascade.network_units import SupervisedUnit
import tensorflow as tf
import datetime

class SingleLayerMLP(SupervisedUnit):
    """

    """
    def __init__(self, loss_function, parameters, name = None):
        super().__init__(loss_function, parameters, name)
        self._hidden_layer_dim = self._parameters["hidden_layer_dim"]

    def initialize_unit(self):

        input_x = self.input_layer.get_data()
        input_y = self.target_layer.get_data()
        # Build the model.
        try: # TODO make helper function out of it
            input_shape = (input_x.shape[0],input_x.shape[1])
        except IndexError:
            input_shape = (input_x.shape[0],1)
        try:
            output_shape = (input_y.shape[0], input_y.shape[1])
        except IndexError:
            output_shape = (input_y.shape[0], 1)

        self.network_model = models.Sequential([
            layers.Dense(self._hidden_layer_dim, activation='sigmoid', input_shape=(input_shape[1],)),
            layers.Dense(output_shape[1], activation='linear' )
        ])

        # Compile the model.
        self.network_model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy'],
        )
        self.initialized = True

    def casscade_train(self):
        x = self.input_layer.get_data()
        y = self.target_layer.get_data()
        self.train(x, y)

    def train(self, input_x, target_y):
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.network_model.fit(
                    input_x,
                    target_y,
                    epochs=25,
                    batch_size=150,
         #   callbacks=[tensorboard_callback]
        )

    def activate(self, x_test):
        output = self.network_model.predict(x_test)
        self.set_output(output)
        return output

    def set_output(self, data):
        if not (self.output_layer is None):
            self.output_layer.set_state(data)
        return data

    def parameters(self):
        return {"hidden_layer_dim": None}