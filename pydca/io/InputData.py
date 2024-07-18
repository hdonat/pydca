#!/usr/bin/python
#-*- coding: utf-8 -*-

from Data import Data

class InputData(Data):
    def __init__(self, data_file_path, layer_levels, F_dpos = 6):
        super(Data, self).__init__(data_file_path, layer_levels, F_dpos)

    def get_data(self):
        return np.delete(self.data_matrix, [self.F_dpos,self.F_dpos+1,self.F_dpos+2], axis=1) # np.append(self.data_matrix[:,:6], self.data_matrix[:,9:])
