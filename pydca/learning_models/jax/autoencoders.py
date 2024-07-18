from pydca.pydca.cascade.network_units import UnsupervisedUnit
import flax
from flax import linen as nn
from typing import Any, Callable, Sequence, Optional

import jax.numpy as np
class MLP(nn.Module):
  features: Sequence[int]

@nn.compact
def __call__(self, x):
    for feat in self.features[:-1]:
      x = flax.nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


class AutoEncoder(UnsupervisedUnit):

    def __init__(self, loss_function):
        super().__init__(loss_function)

    def casscade_train(self):
        pass

    def train(self, input_x, output_y):
        pass

    def activate(self):
        pass

    def set_output(self):
        pass

    def initialize_unit(self):
        pass

    def parameters(self):
        pass