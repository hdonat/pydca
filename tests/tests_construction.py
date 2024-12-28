import pytest
from sklearn.datasets import load_iris
import numpy as np
from pydca.cascade.cascades import CascadeNetwork
from pydca.learning_models.sklearn.simple_regression_models import LinearModel
from pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


class TestCascadeMethods:
    @pytest.fixture
    def iris_data(self):
        data = load_iris()
        return {"data": data["data"], "target": data["target"].reshape(-1, 1)}

    def test_instantiation(self):
        cascade = CascadeNetwork()
        assert isinstance(cascade, CascadeNetwork)

    def test_add_networks(self):
        cascade = CascadeNetwork()
        assert isinstance(cascade, CascadeNetwork)
        cascade.add_cascade_layer(LinearModel(root_mean_squared_error))
        cascade.add_cascade_layer(LinearModel(root_mean_squared_error))

    def test_add_networks_and_plot_graph(self, tmp_path):
        cascade = CascadeNetwork("Test_Add_Two_Layers")
        assert isinstance(cascade, CascadeNetwork)

        cascade.add_cascade_layer(LinearModel(root_mean_squared_error))
        cascade.add_cascade_layer(
            LinearModel(root_mean_squared_error),
        )

        output_dir = tmp_path / "test_results"
        output_dir.mkdir()
        cascade.draw_graph(str(output_dir))

    def test_train_cascade(self, iris_data, tmp_path):
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        assert isinstance(cascade, CascadeNetwork)

        # Add and train networks
        cascade.add_cascade_layer(LinearModel(root_mean_squared_error))
        cascade.add_cascade_layer(LinearModel(root_mean_squared_error))
        cascade.train_networks(iris_data["data"], iris_data["target"], True)

        # Get predictions
        y_hat = cascade.predict(iris_data["data"])

        # Verify predictions shape
        expected_shape = iris_data["target"].shape[0]
        assert (
            y_hat.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {y_hat.shape}"

        # Optional: Save visualization
        output_dir = tmp_path / "test_results"
        output_dir.mkdir()
        plot_predictions(
            iris_data["target"], y_hat, output_dir / "train_two_linear.png"
        )

    def test_train_mlp_cascade(self, iris_data, tmp_path):
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        assert isinstance(cascade, CascadeNetwork)

        # Add and train networks
        cascade.add_cascade_layer(
            MLPModel(root_mean_squared_error, {"hidden_layer_dim": 100})
        )
        cascade.add_cascade_layer(
            MLPModel(root_mean_squared_error, {"hidden_layer_dim": 100})
        )

        cascade.train_networks(iris_data["data"], iris_data["target"], True)

        # Get predictions
        y_hat = cascade.predict(iris_data["data"])

        # Verify predictions shape
        expected_shape = (iris_data["target"].shape[0], 1)
        assert (
            y_hat.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {y_hat.shape}"

        # Optional: Save visualization
        output_dir = tmp_path / "test_results"
        output_dir.mkdir()
        plot_predictions(
            iris_data["target"], y_hat, output_dir / "train_two_single_mlp.png"
        )


def plot_predictions(y_true, y_pred, save_path):
    """Helper function to plot and save predictions visualization."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(y_true.T, c="green", label="True")
    plt.plot(y_pred[:, -1:], c="red", label="Predicted")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
