from dataclasses import dataclass
import os
import time
import numpy as np
import pandas as pd

import collections
from typing import List

import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
 
'''
* ohara-cipa-v1-2017_fixedconc-v1.mmt 
* fixed concentrations
* cell type : Mid-myocardial
'''

def find_closest_index(array, t):
    """Given an array, return the index with the value closest to t."""
    return (np.abs(np.array(array) - t)).argmin()

def get_currents_with_constant_dt(params, x): # avg, min, max
        
    window = params['window']
    step_size = params['step_size']
    window_type = params['window_type']
    
    times = x[0]
    i_ion = x[1]
              
    i_ion_window = []
    t = 0  
    if window_type == 'avg':
        while t <= times[-1] - window:
            start_index = find_closest_index(times, t)
            end_index = find_closest_index(times, t + window)            
            I_window = i_ion[start_index: end_index + 1]                                       
            i_ion_window.append(sum(I_window)/len(I_window))  
            t += step_size     
    if window_type == 'min':
        while t <= times[-1] - window:
            start_index = find_closest_index(times, t)
            end_index = find_closest_index(times, t + window)            
            I_window = i_ion[start_index: end_index + 1]                           
            i_ion_window.append(I_window.min())      
            t += step_size
    elif window_type == 'max':    
        while t <= times[-1] - window:
            start_index = find_closest_index(times, t)
            end_index = find_closest_index(times, t + window)            
            I_window = i_ion[start_index: end_index + 1] 
            i_ion_window.append(I_window.max())                                                  
            t += step_size
    elif window_type == 'amax':
        while t <= times[-1] - window:
            start_index = find_closest_index(times, t)
            end_index = find_closest_index(times, t + window)            
            I_window = i_ion[start_index: end_index + 1]       
            I_window_max = I_window.max()
            I_window_min = I_window.min()
            if abs(I_window_max) > abs(I_window_min):
                i_ion_window.append(I_window_max)                
            else : 
                i_ion_window.append(I_window_min)                                        
            t += step_size
    elif window_type == 'avg_min':
        while t <= times[-1] - window:
            start_index = find_closest_index(times, t)
            end_index = find_closest_index(times, t + window)            
            I_window = i_ion[start_index: end_index + 1]                   
            i_ion_window.append([sum(I_window)/len(I_window), I_window.min()])                        
            t += step_size    
    elif window_type == 'avg_amax_min':
        while t <= times[-1] - window:
            start_index = find_closest_index(times, t)
            end_index = find_closest_index(times, t + window)            
            I_window = i_ion[start_index: end_index + 1]                           
            I_window_amax = None
            I_window_max = I_window.max()
            I_window_min = I_window.min()
            if abs(I_window_max) > abs(I_window_min):
                I_window_amax = I_window_max
            else : 
                I_window_amax = I_window_min                            
            i_ion_window.append([sum(I_window)/len(I_window), I_window_amax, I_window_min])                        
            t += step_size    
    elif window_type == 'all':
        while t <= times[-1] - window:
            start_index = find_closest_index(times, t)
            end_index = find_closest_index(times, t + window)            
            I_window = i_ion[start_index: end_index + 1]       
            I_window_amax = None
            I_window_max = I_window.max()
            I_window_min = I_window.min()
            if abs(I_window_max) > abs(I_window_min):
                I_window_amax = I_window_max
            else : 
                I_window_amax = I_window_min                            
            i_ion_window.append([sum(I_window)/len(I_window), I_window_amax, I_window_min, I_window_max])                        
            t += step_size
            
    return i_ion_window




def get_data(params, fileNo=0):
    dataset_dir = os.path.dirname(os.path.realpath(__file__))     
    xs = np.load(os.path.join(dataset_dir, 'currents%d.npy'%(fileNo) ) , allow_pickle=True)
    ys = np.load(os.path.join(dataset_dir, 'parameter%d.npy'%(fileNo) ) )
    return xs, ys

# This method is useful of a case that the number of dataset is small or you want to raw data
def get_dataset(file_numbers, window=10, step_size=5, multi=False, torch_tensor=False):
    start_time = time.time()

    processes = len(file_numbers)
    if len(file_numbers)>36:
        processes = 36  

    nCPU = os.cpu_count()  
    if nCPU>36:
        nCPU = 36  

    params = {
        'window' : window,
        'step_size' : step_size,        
    }    

    xs_li = []
    ys_li = []
    if torch_tensor:
        print("dddd")        
        # if multi and len(file_numbers)>1:                
        #     pool = multiprocessing.Pool(processes=processes)
        #     func = partial(get_data, None)
        #     dataset_li = pool.map(func, file_numbers)
        #     pool.close()
        #     pool.join()  
        #     for i, dataset in enumerate(dataset_li):            
        #         xs_li.append(torch.tensor(dataset[0]))
        #         ys_li.append(torch.tensor(dataset[1]))
        # else :
        #     for i, fileNo in enumerate(file_numbers):
        #         xs, ys = get_data(None, fileNo)   
        #         xs_li.append(torch.tensor(xs))
        #         ys_li.append(torch.tensor(ys))

        # xs_li = torch.cat(xs_li)
        # ys_li = torch.cat(ys_li)   
        
        # # pool = multiprocessing.Pool(processes=nCPU)
        # # func = partial(get_currents_with_constant_dt, params)
        # # xs_li = np.array(pool.map(func, xs_li))
        # # pool.close()
        # # pool.join() 

    else :          
        if multi and len(file_numbers)>1:                            
            pool = multiprocessing.Pool(processes=processes)
            func = partial(get_data, None)
            dataset_li = pool.map(func, file_numbers)
            pool.close()
            pool.join()              
            for dataset in tqdm(dataset_li):        
                xs_li.append(dataset[0])    
                ys_li.append(dataset[1])
        else :            
            for fileNo in tqdm(file_numbers):                
                xs, ys = get_data(None, fileNo)    
                xs_li.append(xs)
                ys_li.append(ys)          

        xs_li = np.concatenate( xs_li, axis=0)
        ys_li = np.concatenate( ys_li, axis=0)
        
        if window>0 and step_size>0:
            pool = multiprocessing.Pool(processes=nCPU)
            func = partial(get_currents_with_constant_dt, params)
            xs_li = np.array(pool.map(func, xs_li))
            pool.close()
            pool.join()  

    print("--- %s seconds ---"%(time.time()-start_time))
    return xs_li , ys_li



def get_data2(params, fileNo=0):
    dataset_dir = os.path.dirname(os.path.realpath(__file__))     
    xs = np.load(os.path.join(dataset_dir, 'currents%d.npy'%(fileNo) ) , allow_pickle=True)
    ys = np.load(os.path.join(dataset_dir, 'parameter%d.npy'%(fileNo) ) )

    if params['noise_sigma'] != None and params['noise_sigma']>0:
        for x in xs:            
            x[1] = x[1] + np.random.normal(0, params['noise_sigma'], x[1].shape) # add noise

    if params['window']>0 and params['step_size']>0:
        temp_li = []

        for x in xs:
            temp = get_currents_with_constant_dt(params=params, x=x)
            temp_li.append(temp)

        return np.array(temp_li), ys
    else :
        return xs, ys

# This method is useful of a case that the number of dataset is large
def get_dataset2(file_numbers, window=10, step_size=5, window_type='avg', noise_sigma=0, multi=False, torch_tensor=False):             
    start_time = time.time()

    processes = len(file_numbers)
    if len(file_numbers)>os.cpu_count():
        processes = os.cpu_count()   

    params = {
        'window' : window,
        'step_size' : step_size,   
        'window_type' : window_type,
        'noise_sigma' : noise_sigma,        
    }  

    xs_li = []
    ys_li = []
    if torch_tensor:        
        if multi and len(file_numbers)>1:                
            pool = multiprocessing.Pool(processes=processes)
            func = partial(get_data2, params)
            dataset_li = pool.map(func, file_numbers)
            pool.close()
            pool.join()  
            for i, dataset in enumerate(dataset_li):            
                xs_li.append(torch.tensor(dataset[0]))
                ys_li.append(torch.tensor(dataset[1]))
        else :
            for fileNo in tqdm(file_numbers):
                xs, ys = get_data2(params, fileNo)   
                xs_li.append(torch.tensor(xs))
                ys_li.append(torch.tensor(ys))

        xs_li = torch.cat(xs_li)
        ys_li = torch.cat(ys_li)              

    else :          
        if multi and len(file_numbers)>1:                            
            pool = multiprocessing.Pool(processes=processes)
            func = partial(get_data2, params)
            dataset_li = pool.map(func, file_numbers)
            pool.close()
            pool.join()              
            for dataset in tqdm(dataset_li):        
                xs_li.append(dataset[0])    
                ys_li.append(dataset[1])
        else :            
            for fileNo in tqdm(file_numbers):                
                xs, ys = get_data2(params, fileNo)    
                xs_li.append(xs)
                ys_li.append(ys)          

        xs_li = np.concatenate( xs_li, axis=0)
        ys_li = np.concatenate( ys_li, axis=0)

    print("--- %s seconds ---"%(time.time()-start_time))
    return xs_li , ys_li




