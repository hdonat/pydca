import unittest
from pydca.pydca.cascade.cascades import CascadeNetwork
from pydca.pydca.learning_models.sklearn.simple_regression_models import LinearModel
from pydca.pydca.learning_models.keras.autoencoders import AutoEncoder
from pydca.pydca.cascade.filters import Filter
from pydca.pydca.learning_models.keras.mlps import SingleLayerMLP
from pydca.pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
import sklearn.datasets

class TestCascadeMethods(unittest.TestCase):

    def test_instantiation(self):
        cascade = CascadeNetwork()
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)

    def test_add_networks(self):
        cascade = CascadeNetwork()
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(LinearModel(mean_squared_error))
        cascade.add_network_layer(LinearModel(mean_squared_error))

    def test_add_networks_and_plot_graph(self):
        cascade = CascadeNetwork("Test_Add_Two_Layers")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(LinearModel(mean_squared_error))
        cascade.add_network_layer(LinearModel(mean_squared_error), filter_list=[Filter()])
        cascade.draw_graph("test_results")

    def test_supervised_unsupervised_and_plot_graph(self):
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(LinearModel(mean_squared_error))
        cascade.add_network_layer(AutoEncoder(mean_squared_error, {"encoding_dim": 32}), connect_to_inlayer=0)
        cascade.draw_graph("test_results")

    def test_large_supervised_unsupervised_and_plot_graph(self):
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade_Large")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(LinearModel(mean_squared_error))
        cascade.add_network_layer(AutoEncoder(mean_squared_error, {"encoding_dim": 32}),connect_to_inlayer=0)
        cascade.add_network_layer(LinearModel(mean_squared_error))
        cascade.add_network_layer(AutoEncoder(mean_squared_error, {"encoding_dim": 32}),connect_to_inlayer=0)
        cascade.draw_graph("test_results")


    def test_train_cascade(self):
        data = sklearn.datasets.load_iris()
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(LinearModel(mean_squared_error))
        cascade.add_network_layer(LinearModel(mean_squared_error))
        target = data["target"].reshape(-1, 1)
        cascade.train_networks(data["data"],target, True)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (data["target"].shape[0], data["target"].shape[1])
        except IndexError:
            shape = (data["target"].shape[0],1)
        plt.plot(data["target"].T, c="green")
        plt.plot(y_hat[:,-shape[1]:], c="red")
        plt.savefig("test_results/train_two_linear.png")


    def test_train_mlp_cascade(self):
        data = sklearn.datasets.load_iris()
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": 100}))
        cascade.add_network_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": 100}))
        target = data["target"].reshape(-1, 1)
        cascade.train_networks(data["data"],target, True)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (data["target"].shape[0], data["target"].shape[1])
        except IndexError:
            shape = (data["target"].shape[0],1)
        plt.plot(data["target"].T, c="green")
        plt.plot(y_hat[:,-shape[1]:], c="red")
        plt.savefig("test_results/train_two_single_mlp.png")


if __name__ == '__main__':
    unittest.main()
