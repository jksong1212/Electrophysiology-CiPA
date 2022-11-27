import os, copy
import numpy as np
import pandas as pd

import multiprocessing
from functools import partial

import torch
 
'''


'''

def get_times(scale=1):   
    dataset_dir = os.path.dirname(os.path.realpath(__file__)) 
    df = pd.read_csv( os.path.join(dataset_dir, 'herg25oc1-staircaseramp-times.csv'))    
    return df['time'].to_numpy()[::scale]

def get_data(scale, cell):
    dataset_dir = os.path.dirname(os.path.realpath(__file__))     
    current_filename = 'herg25oc1-staircaseramp-%s.csv'%(cell)    
    parameter_filename = 'herg25oc1-staircaseramp-%s-solution-542811797.txt'%(cell)

    I = pd.read_csv( os.path.join(dataset_dir, current_filename)).to_numpy()    
    P = pd.read_csv( os.path.join(dataset_dir, parameter_filename), header=None).to_numpy()
    I = I[::scale]
    I = I.reshape( 1, len(I) )
    P = P.reshape( 1, len(P) )
    
    return I, P, cell
    
def get_dataset(cell_to_remove=[], scale=1, multi=False, torch_tensor=False):        
    dataset_dir = os.path.dirname(os.path.realpath(__file__)) 
    full_cell_li = []
    file_list = os.listdir(dataset_dir)
    param_list = [file for file in file_list if file.endswith("542811797.txt")]

    for file in param_list:
        full_cell_li.append(file[24:27]) 

    nRemovedCell = 0
    print("The number of full cells :", len(full_cell_li))
    cell_li = copy.copy(full_cell_li)
    for i, cell in enumerate(full_cell_li):
        if cell in cell_to_remove:
            cell_li.remove(cell)
            # print("%d : %s is removed"%(i+1, cell))
            nRemovedCell += 1
    print("The number of removed Cells :", nRemovedCell)
    print("The number of cells :", len(cell_li))
    # print(cell_li)
        
    processes = len(cell_li)
    if len(cell_li)>os.cpu_count():
        processes = os.cpu_count()

    if torch_tensor:
        currents = []
        parameters = []
        cells = []
        if multi and len(cell_li)>1:                
            pool = multiprocessing.Pool(processes=processes)
            func = partial(get_data, scale)
            dataset_li = pool.map(func, cell_li)
            pool.close()
            pool.join()  

            for i, dataset in enumerate(dataset_li):            
                currents.append(torch.tensor(dataset[0]))
                parameters.append(torch.tensor(dataset[1]))
                cells.append(dataset[2])
        else :
            for i, cell in enumerate(cell_li):
                I, P, C = get_data(scale, cell)   
                currents.append(torch.tensor(I))
                parameters.append(torch.tensor(P))
                cells.append(C)
        return torch.cat(currents), torch.cat(parameters), cells

    else : 
        currents = None
        parameters = None
        cells = []
        if multi and len(cell_li)>1:                
            pool = multiprocessing.Pool(processes=processes)
            func = partial(get_data, scale)
            dataset_li = pool.map(func, cell_li)
            pool.close()
            pool.join()  

            for i, dataset in enumerate(dataset_li):            
                if i==0:
                    currents = dataset[0]
                    parameters = dataset[1]                
                else:
                    currents = np.concatenate( (currents, dataset[0]), axis=0)
                    parameters = np.concatenate( (parameters, dataset[1]), axis=0)    
                cells.append(dataset[2])
        else :
            for i, fileNo in enumerate(cell_li):
                I, P, C = get_data(scale, fileNo)    
                # print(I.shape, P.shape)               
                if i==0:
                    currents = I
                    parameters = P
                else:
                    currents = np.concatenate( (currents, I), axis=0)
                    parameters = np.concatenate( (parameters, P), axis=0)
                cells.append(C)
        return np.array(currents) , np.array(parameters), cells
        
        
