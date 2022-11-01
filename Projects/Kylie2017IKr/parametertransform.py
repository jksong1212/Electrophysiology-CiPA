#!/usr/bin/env python
import numpy as np


def log_transform_from_model_param(param):
    # Apply natural log transformation to all parameters
    out = np.copy(param)
    out = np.log(out)
    return out


def log_transform_to_model_param(param):
    # Inverse of log_transform_from_model_param()
    # Apply natural exp transformation to all parameters
    out = np.copy(param)
    out = np.exp(out)
    return out


def donothing(param):
    out = np.copy(param)
    return out

            
            
# Parameter range setting
dataset_dir = 'Kylie'
parameter_ranges = []
if 'Kylie' in dataset_dir:    
    if 'rmax600' in dataset_dir:
        # Kylie: rmax = 600 
        parameter_ranges = []
        parameter_ranges.append( [100, 500000] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 260] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 130] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 260] )
        parameter_ranges.append( [0.0001, 598] )
        parameter_ranges.append( [0.0001, 130] )
        print("Kylie-rmax600 dataset has been selected.")
    else :
        # Kylie
        parameter_ranges.append([100, 500000])
        parameter_ranges.append( [0.0001, 1000000])
        parameter_ranges.append( [0.0001, 384])
        parameter_ranges.append( [0.0001, 1000000] )
        parameter_ranges.append( [0.0001, 192] )
        parameter_ranges.append( [0.0001, 1000000] )
        parameter_ranges.append( [0.0001, 384] )
        parameter_ranges.append( [0.0001, 1000000] )
        parameter_ranges.append( [0.0001, 192] )
        print("Kylie dataset has been selected.")

elif 'RealRange' in dataset_dir:
        parameter_ranges.append([3134, 500000])                 # g
        parameter_ranges.append( [0.0001, 2.6152843264828003])  # p1
        parameter_ranges.append( [43.33271226094526, 259])      # p2
        parameter_ranges.append( [0.001, 0.5] )                 # p3
        parameter_ranges.append( [15, 75] )                     # p4
        parameter_ranges.append( [0.8, 410] )                   # p5
        parameter_ranges.append( [0.0001, 138.] )               # p6
        parameter_ranges.append( [1.0, 59] )                    # p7
        parameter_ranges.append( [1.6, 90] )                    # p8
        print("RealRange dataset has been selected.")

parameter_ranges = np.array(parameter_ranges)
print(parameter_ranges.shape)


def transform_to_norm(param):
    # Apply natural log transformation to all parameters
    param_names = ['g', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    out = np.copy(param)
    for i, p_name in enumerate(param_names):
        if p_name in ['p1', 'p3', 'p5', 'p7']:        
            out[i] = ( np.log(out[i]) - np.log(parameter_ranges[i,0]) ) / ( np.log(parameter_ranges[i,1]) - np.log(parameter_ranges[i,0]) )
        else:
            out[i] = (out[i] - parameter_ranges[i,0]) / (parameter_ranges[i,1] - parameter_ranges[i,0])

    return out


def transform_to_original(param):
    # Inverse of log_transform_from_model_param()
    # Apply natural exp transformation to all parameters
    param_names = ['g', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    out = np.copy(param)
    for i, p in enumerate(['g', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']):
        if p in ['p1', 'p3', 'p5', 'p7']:
            out[i] = np.exp( out[i] * (np.log(parameter_ranges[i,1])-np.log(parameter_ranges[i,0])) + np.log(parameter_ranges[i,0]) )                
        else:
            out[i] = out[i] * (parameter_ranges[i,1]-parameter_ranges[i,0]) + parameter_ranges[i,0]

    return out