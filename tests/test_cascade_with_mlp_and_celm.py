import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from pydca.cascade.cascades import CascadeNetwork
from pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel
from pydca.learning_models.my_models.celm import CELM


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class TestCascadeNetworks:
    @pytest.fixture
    def simple_sine_data(self):
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + np.random.random_sample(x.shape)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return train_test_split(x, y, test_size=0.25)

    @pytest.fixture
    def complex_sine_data(self):
        x = np.linspace(0, 2 * np.pi, 500)
        y = (
            np.sin(x)
            + np.cos(x) ** 2
            + np.sin(x) ** 3
            + (np.random.random_sample(x.shape) - 0.5)
        )
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return train_test_split(x, y, test_size=0.25)

    def test_mlp_cascade(self, simple_sine_data):
        x_train, x_test, y_train, y_test = simple_sine_data
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")

        # Add cascade layers
        cascade.add_cascade_layer(
            MLPModel(mean_squared_error, {"hidden_layer_dim": (1,)})
        )
        cascade.add_cascade_layer(
            MLPModel(mean_squared_error, {"hidden_layer_dim": (1,)})
        )
        cascade.add_cascade_layer(
            MLPModel(mean_squared_error, {"hidden_layer_dim": (100,)})
        )

        # Train and predict
        cascade.train_networks(x_train, y_train, True)
        y_hat = cascade.predict(x_test, 1)

        # Assert predictions are reasonable
        mse = mean_squared_error(y_test.flatten(), y_hat.flatten())
        assert mse < 1.0, f"MSE {mse} is too high"

    def test_celm_cascade(self, simple_sine_data):
        x_train, x_test, y_train, y_test = simple_sine_data
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")

        # Add CELM layer
        cascade.add_cascade_layer(
            CELM(mean_squared_error, {"hidden_size": 100, "constr": {}})
        )

        # Train and predict
        cascade.train_networks(x_train, y_train, True)
        y_hat = cascade.predict(x_test, 1)

        # Assert predictions are reasonable
        mse = mean_squared_error(y_test.flatten(), y_hat.flatten())
        assert mse < 1.0, f"MSE {mse} is too high"

    def test_incremental_cascade(self, complex_sine_data):
        x_train, x_test, y_train, y_test = complex_sine_data
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")

        for i in range(3):
            cascade.add_cascade_layer(
                MLPModel(mean_squared_error, {"hidden_layer_dim": (1,)})
            )
            cascade.train_networks(x_train, y_train, True)

            # Get predictions and calculate error
            y_hat = cascade.predict(x_train, 1)

            # Update training data with high-error samples
            score = ((y_train - y_hat) ** 2).flatten()
            mean_error = np.mean(score)
            indexes = np.where(score > mean_error)
            x_train = np.vstack([x_train, x_train[indexes]])
            y_train = np.vstack([y_train, y_train[indexes]])

        # Final test predictions
        y_hat = cascade.predict(x_test, 1)
        test_error = mean_squared_error(y_test.flatten(), y_hat.flatten())
        assert test_error < 1.0, f"Test error {test_error} is too high"
