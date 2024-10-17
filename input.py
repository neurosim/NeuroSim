# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:43:52 2020

@author: yando
"""
import csv
from utils import is_integer, is_float, convert_row, create_dict


# read the input .csv file
def get_input(file_name="VGG8_ws_256x256.csv"):
    
    var_dict_list = [] # store the inputs for each layer in a dict
    
    with open(file_name, 'r') as file:
        rows = csv.reader(file, delimiter=',')
        header = [] 
        for i, row in enumerate(rows):
            if i == 0:
                header = row
            else:
                row = convert_row(row)
                var_dict = create_dict(header, row)
                var_dict_list.append(var_dict)
    
    return var_dict_list
    