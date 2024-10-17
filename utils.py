# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 08:47:36 2020

@author: yando
"""

import re

def is_integer(string):
    
    # determine whether the string constains integer or not
    # return True or False
    
    string = str(string).strip().lstrip('-').lstrip('+')
    if string.isnumeric() == True:
        return True
    else:
        return False
    
def is_float(string):
    
    # determine whether the string constains floating point number or not
    # return True or False    
    # reference: https://blog.mimvp.com/article/25955.html
    
    string = string.strip().lstrip('-').lstrip('+')
    if string.isnumeric() == False:
        try:
            float(string)
            return True
        except Exception as ex:
            # print("is_float() - error: " + str(ex))
            return False
    else:
        return False

def convert_row(csv_row):
    for i in range(len(csv_row)):
        if is_integer(csv_row[i]) == True:
            csv_row[i] = int(csv_row[i])
        elif is_float(csv_row[i]) == True:
            csv_row[i] = float(csv_row[i])
    
    return csv_row

def create_dict(key, value):
    # two lists key and value as input
    dictionary = dict(zip(key, value))
    
    return dictionary