import keras
from keras import layers
from pydca.pydca.cascade.network_units import UnsupervisedUnit

class AutoEncoder(UnsupervisedUnit):
    """

    """
    def __init__(self, loss_function, parameters, name = None):
        super().__init__(loss_function, parameters, name)
        self._encoding_dim = parameters["encoding_dim"]

    def initialize_unit(self):
        input_x = self.input_layer.get_data()
        self._input_layer = keras.Input(shape=input_x.shape[1])
        encoded = layers.Dense(self._encoding_dim, activation='tanh')(self._input_layer)
        decoded = layers.Dense(input_x.shape[1], activation='sigmoid')(encoded)
        self._encoder = keras.Model(self._input_layer, encoded)
        self._network_model = keras.Model(self._input_layer, decoded)

        encoded_input = keras.Input(shape=(self._encoding_dim,))
        # Retrieve the last layer of the autoencoder model
        decoder_layer = self._network_model.layers[-1]
        # Create the decoder model
        self._decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

        self._network_model.compile(optimizer='adam', loss='mean_squared_error')
        self.initialized = True

    def casscade_train(self):
        x = self.input_layer.get_data()
        self.train(x)

    def train(self, input_x):
        self._network_model.fit(input_x, input_x,
                        epochs=self._parameters["epochs"],
                        batch_size=256,
                        shuffle=True,
                        validation_data=(input_x, input_x), verbose="auto")

    def activate(self, x_test):
        encoded_data = self._encoder.predict(x_test)
        decoded_imgs = self._decoder.predict(encoded_data)
        return self.set_output(encoded_data)

    def set_output(self, data):
        if not (self.output_layer is None):
            self.output_layer.set_state(data)
        return data

    def parameters(self):
        return {"encoder_weights": None}