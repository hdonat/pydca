#!/usr/bin/python
# -*- coding: utf-8 -*-

from Data import Data
from SaveableData import SaveableData


class OutputData(Data, SaveableData):
    def __init__(self, data_file_path, layer_levels, F_dpos=6):
        super(Data, self).__init__(data_file_path, layer_levels, F_dpos)

    def get_data(
        self,
    ):
        return self.data_matrix[:, self.F_dpos : self.F_dpos + 3]

    def save(
        self,
    ):
        pass
