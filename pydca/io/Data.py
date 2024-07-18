#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import normalize
import pdb

class Data:
    def __init__(self, data_or_file_path,  F_dpos = 6, F_size = 3, null_end=False, skip_header_num=1, layer_levels = 10):
        if getattr(data_or_file_path, "shape", False):
            self.data_matrix = data_or_file_path
        else:
            self.data_matrix = np.genfromtxt(data_or_file_path + '.csv', delimiter=',', skip_header = skip_header_num)
            if null_end:
                self.data_matrix = self.data_matrix[:,:-1]
        self.F_dpos = F_dpos # position of the 3 force values in every row
        self.F_size = F_size
        self.layer_levels = layer_levels
        self.input_norm = None
        self.output_norm = None
        self.sub_norms = {}

    def get_data(self):
        """
        Implemented but unnessacary
        """
        return self.data_matrix

    def create_random_data_set(self, count_of_splits):
        sample_sets = []
        max_index = self.data_matrix.shape[0]
        all_indexes = np.array(range(max_index))
        subset_size = max_index/count_of_splits
        for i in range(count_of_splits):
            rand_idx = np.random.randint(all_indexes.size, size=subset_size)
            subset_idx = all_indexes[rand_idx]
            sub_data_matrix = self.data_matrix[subset_idx,:]
            sample_sets.append(Data(sub_data_matrix, self.layer_levels, self.F_dpos, self.F_size))
            all_indexes = np.delete(all_indexes, rand_idx)
        return sample_sets


    def get_input_data(self):
#        result, self.input_norm = normalize(np.delete(self.data_matrix, [6,7,8], axis=1), return_norm=True) # np.append(self.data_matrix[:,:6], self.data_matrix[:,9:])
        result = np.delete(self.data_matrix, list(range(self.F_dpos,self.F_size+self.F_dpos)), axis=1)
        return result

    def get_output_data(self):
#        result, self.output_norm =  normalize(self.data_matrix[:,6:9], return_nJonas Cleveorm=True)
        result = self.data_matrix[:,self.F_dpos:self.F_size+self.F_dpos]
        return result

    def get_input_data_for_level(self, level):
        if level < self.layer_levels:
            data = self.get_input_data()
            #result,norm =  normalize(np.append(data[:,:7], data[:,7::(self.layer_levels - level)], axis=1), return_norm=True)
            result =  np.append(data[:,:7], data[:,7::(self.layer_levels - level)], axis=1)
            #self.sub_norms[level] = norm
            return result

    def get_input_data_for_level_extern(self, data, level):
        if level < self.layer_levels:
            result =  np.append(data[:,:7], data[:,7::(self.layer_levels - level)], axis=1)
            return result

    def denormalize_indata(self, data):
        """
        TODO STUB
        """
        return data*self.input_norm.T

    def denormalize_outdata(self, data):
        """
        TODO STUB
        """
        return data*self.output_norm.T


    def denormalize_subdata(self, data, level):
        """
        TODO STUB
        """
        return data*self.sub_norms[level].T
