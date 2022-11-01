"""Contains classes to store the result of a genetic algorithm run.

Additionally, the classes in this module allow for figure generation.
"""

from abc import ABC
import copy
import enum
import math
import random
from typing import Dict, List, Union
from os import listdir, mkdir

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

import ga_configs
import mod_protocols as protocols
import mod_trace as trace
import mod_kernik as kernik


#############################################
from scipy.integrate import ode, solve_ivp
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bisect

sys.path.append('../')
sys.path.append('../Protocols')
import simulator_scipy

# Models
from Models.br1977 import BR1977
from Models.ord2011JK_v1 import ORD2011
#############################################



def get_model_response_JK( model, protocol, prestep=None):    
    
    simulator = simulator_scipy.Simulator(model)     
    
    if prestep == None:
        print("There is no pre-step simulation.")
    elif prestep == 5000:        
        y0 = [-8.69999996e+01,  6.94732336e+00,  6.94736848e+00,  1.44992431e+02,
               1.44992434e+02,  5.48328391e-05,  5.40431668e-05,  1.25617506e+00,
               1.25618638e+00,  8.12231733e-03,  6.62326077e-01,  6.62326077e-01,
               6.62326077e-01,  4.14582271e-01,  6.62326077e-01,  2.27721811e-04,
               4.79645030e-01,  2.87189165e-01,  1.07103663e-03,  9.99468797e-01,
               9.99468797e-01,  5.45740810e-04,  9.99468797e-01,  9.99468797e-01,
               2.96634937e-09,  9.99999988e-01,  9.99999988e-01,  9.99999988e-01,
               9.99999988e-01,  9.99999988e-01,  4.78979614e-04,  9.99999988e-01,
               9.99999988e-01,  9.28750206e-06,  9.23466020e-06,  1.96054631e-04,
               2.15667189e-04,  9.97012407e-01,  1.27419629e-07,  1.59274616e-07,
               2.47073549e-04]
        simulator.model.y0 = y0        
    else:
        simulator.pre_simulate( pre_step=5000, protocol='constant' )
    
    solution = simulator.simulate( [0, protocol.get_voltage_change_endpoints()[-1]] , max_step=0.5, atol=1e-08, rtol=1e-10) 
    
    command_voltages = [protocol.get_voltage_at_time(t) for t in solution.t]

    # print("dd11",len(model.current_response_info.currents))
    
    simulator.model.current_response_info = trace.CurrentResponseInfo()
    if len(solution.y) < 200:
        list(map(simulator.model.differential_eq, solution.t, solution.y.transpose()))
    else:
        list(map(simulator.model.differential_eq, solution.t, solution.y))               
    
    # print("dd22",len(model.current_response_info.currents))
    
    tr = trace.Trace(protocol,
                     cell_params=None,
                     t=solution.t,
                     y=command_voltages,  # simulator.model.V,
                     command_voltages=command_voltages,
                     current_response_info=model.current_response_info,
                     default_unit=None)
    
    return tr



def get_model_response_JK2( model, protocol, prestep=None):    
    
    simulator = simulator_scipy.Simulator(model)     
    
    if prestep == None:
        print("There is no pre-step simulation.")
    elif prestep == 'pre':
        print("There is no pre-step simulation.")
    else:
        simulator.pre_simulate( pre_step=prestep, protocol='constant' )
    
    times = np.linspace(0, protocol.get_voltage_change_endpoints()[-1], 2000)  
    solution = simulator.simulate2( times=times , max_step=0.5, atol=1e-08, rtol=1e-10) 
        
    command_voltages = [protocol.get_voltage_at_time(t) for t in times]

    # print("dd11",len(model.current_response_info.currents))
    
    simulator.model.current_response_info = trace.CurrentResponseInfo()
    if len(simulator.solver.y) < 200:
        list(map(simulator.model.differential_eq, times, solution.transpose()))
    else:
        list(map(simulator.model.differential_eq, times, solution.y))               
    
    # print("dd22",len(model.current_response_info.currents))
    
    tr = trace.Trace(protocol,
                     cell_params=None,
                     t=times,
                     y=command_voltages,  # simulator.model.V,
                     command_voltages=command_voltages,
                     current_response_info=model.current_response_info,
                     default_unit=None)
    
    return tr