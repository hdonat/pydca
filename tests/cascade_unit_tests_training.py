import unittest
from pydca.pydca.cascade.cascades import CascadeNetwork
from pydca.pydca.learning_models.sklearn.simple_regression_models import LinearModel
from pydca.pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel
from pydca.pydca.learning_models.keras.autoencoders import AutoEncoder
from pydca.pydca.cascade.filters import ZScoreFilter
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
import sklearn.datasets


class TestCascadeMethods(unittest.TestCase):
    def none_test_train_cascade(self):
        data = sklearn.datasets.load_iris()
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(LinearModel(root_mean_squared_error))
        cascade.add_network_layer(LinearModel(root_mean_squared_error))
        target = data["target"].reshape(-1, 1)
        cascade.train_networks(data["data"], target, True)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (data["target"].shape[0], data["target"].shape[1])
        except IndexError:
            shape = (data["target"].shape[0], 1)
        plt.plot(data["target"].T, c="green")
        plt.plot(y_hat[:, -shape[1] :], c="red")
        plt.savefig("test_results/train_two_linear.png")

    def test_train_mlp_cascade(self):
        x, y = make_regression()
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.25)
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        init_scalar = ZScoreFilter("first")
        first_layer_scalar = ZScoreFilter("second")
        cascade.add_network_layer(
            MLPModel(root_mean_squared_error, {"hidden_layer_dim": x.shape[1]}),
            first_layer_filter=init_scalar,
            filter=first_layer_scalar,
        )
        second_layer_scalar = ZScoreFilter("third")
        cascade.add_network_layer(
            MLPModel(root_mean_squared_error, {"hidden_layer_dim": x.shape[1] * 2}),
            filter=second_layer_scalar,
        )
        third_layer_scalar = ZScoreFilter("forth")
        cascade.add_network_layer(
            MLPModel(root_mean_squared_error, {"hidden_layer_dim": x.shape[1] * 3}),
            filter=third_layer_scalar,
        )
        target = y
        cascade.train_networks(x, target, True)
        cascade.apply_data(x_test)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (y.shape[0], y.shape[1])
        except IndexError:
            shape = (y.shape[0], 1)
        y_hat = cascade.layers[-1]._filter.out_filter(y_hat[:, -shape[1] :])
        plt.plot(y_test.T, c="green")
        plt.plot(y_hat, c="red")
        plt.savefig("test_results/train_two_single_mlp.png")

    def test_train_mlp_and_autoencoder_cascade(self):
        x, y = make_regression()
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.25)
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)

        cascade.add_network_layer(
            AutoEncoder(root_mean_squared_error, {"encoding_dim": 20}),
            connect_to_inlayer=0,
            first_layer_filter=ZScoreFilter(),
            filter=ZScoreFilter(),
        )
        cascade.add_network_layer(
            MLPModel(root_mean_squared_error, {"hidden_layer_dim": x.shape[1] * 2}),
            filter=ZScoreFilter(),
        )
        cascade.add_network_layer(
            AutoEncoder(root_mean_squared_error, {"encoding_dim": 40}),
            connect_to_inlayer=0,
            filter=ZScoreFilter(),
        )
        cascade.add_network_layer(
            MLPModel(root_mean_squared_error, {"hidden_layer_dim": x.shape[1] * 3}),
            filter=ZScoreFilter(),
        )
        target = y
        cascade.train_networks(x, target, True)
        cascade.apply_data(x_test)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (y.shape[0], y.shape[1])
        except IndexError:
            shape = (y.shape[0], 1)
        y_hat = cascade.layers[-1]._filter.out_filter(y_hat[:, -shape[1] :])
        plt.plot(y_test.T, c="green")
        plt.plot(y_hat, c="red")
        plt.savefig("test_results/test_train_mlp_and_autoencoder_cascade.png")


if __name__ == "__main__":
    unittest.main()
