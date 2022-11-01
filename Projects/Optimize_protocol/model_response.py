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
import mod_protocols
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
import protocol_lib
import simulator_scipy
import simulator_myokit
import myokit
from Models.br1977 import BR1977
from Models.ord2011 import ORD2011
import mod_trace


def get_model_response_JK( model, protocol, prestep=None):    
        
    simulator = simulator_scipy.Simulator(model)     
            
    if isinstance(model, ORD2011):
        simulator.model.change_cell(1)
        
        if prestep == None:
            print("There is no pre-step simulation.")
        elif prestep == 5000:        
            y0 = [-8.00000002e+01,  6.94549002e+00,  6.94553614e+00,  1.44766826e+02,
                    1.44766919e+02,  5.46283839e-05,  5.38550923e-05,  1.25377967e+00,
                    1.25388387e+00,  1.63694063e-02,  3.83078124e-01,  3.83078124e-01,
                    3.83078124e-01,  1.83137288e-01,  3.83078124e-01,  8.60298196e-04,
                    2.65750283e-01,  1.36776226e-01,  1.71654793e-03,  9.98192733e-01,
                    9.98192733e-01,  8.74934836e-04,  9.98192733e-01,  9.98192733e-01,
                    1.55207580e-08,  9.99999920e-01,  9.99999921e-01,  9.99999920e-01,
                    9.99999920e-01,  9.99999920e-01,  4.72523640e-04,  9.99999920e-01,
                    9.99999920e-01,  2.60425715e-05,  2.54956246e-05,  4.27865719e-04,
                    4.72094402e-04,  9.98307893e-01,  1.12127824e-01,  6.06464622e-07,
                    7.58083393e-07,  2.45431980e-04]
            simulator.model.y0 = y0        
        else:        
            simulator.pre_simulate( protocol='constant', pre_step=prestep, v0=-80)    
    else :
        if prestep == None:
            print("There is no pre-step simulation.")            
        else:        
            simulator.pre_simulate( protocol='constant', pre_step=prestep, v0=-80)  
    
    sol = simulator.simulate( [0, protocol.get_voltage_change_endpoints()[-1]], method='BDF', max_step=1, atol=1e-06, rtol=1e-6)     
    
    command_voltages = [protocol.get_voltage_at_time(t) for t in sol.t]    
    
    if model.is_exp_artefact:
        y_voltages = sol.y[0, :]
    else:
        y_voltages = command_voltages

    tr = trace.Trace(protocol,
                     cell_params=None,
                     t=sol.t,
                     y=y_voltages, 
                     command_voltages=command_voltages,
                     current_response_info=simulator.model.current_response_info,
                     default_unit=None)        
    # print(solution)
    return tr

def get_model_response_with_myokit( model_path, protocol, prestep=None):    
            
    model, p, s = myokit.load(model_path)        
    sim = simulator_myokit.Simulator(model, protocol, max_step=1.0, abs_tol=1e-06, rel_tol=1e-6, vhold=-80)  # 1e-12, 1e-14  # 1e-08, 1e-10
    sim.name = "ohara2017"  
    f = 1.5
    params = {         
        'cell.mode': 2, # Mid-myocardial
        'setting.simType': 1,   # 0: AP   |  1: VC  
        
        'ina.gNa' : 75.0 * f,   
        'inal.gNaL' : 0.0075 * 2.661 * f,  
        'ito.gto' : 0.02 * 4 * f,
        'ical.PCa' : 0.0001 * 1.007 * 2.5 * f,
        'ikr.gKr' : 4.65854545454545618e-2 * 1.3 * f, # [mS/uF]
        'iks.gKs' : 0.0034 * 1.87 * 1.4 * f,
        'ik1.gK1' : 0.1908 * 1.698 * 1.3 * f,
        'inaca.gNaCa' : 0.0008 * 1.4,
        'inak.PNaK' : 30 * 0.7,
        'ikb.gKb' : 0.003,
        'inab.PNab' : 3.75e-10,
        'icab.PCab' : 2.5e-8,
        'ipca.GpCa' : 0.0005,
        
        'ina.g_adj' : 1,  
        'inal.g_adj' : 1,
        'ito.g_adj' : 1,
        'ical.g_adj' : 1,
        'ikr.g_adj' : 1,
        'iks.g_adj' : 1,
        'ik1.g_adj' : 1,
        'inaca.g_adj' : 1,
        'inak.g_adj' : 1,
        'ikb.g_adj' : 1,
        'inab.g_adj' : 1,
        'icab.g_adj' : 1,
        'ipca.g_adj' : 1, 
    }
    sim.set_simulation_params(params)
    
    extra_log = ['ina.INa', 'inal.INaL', 'ito.Ito', 'ical.ICaL', 'ical.ICaNa', 'ical.ICaK', 'ikr.IKr', 'iks.IKs', 'ik1.IK1', 'inaca.INaCa', 'inacass.INaCa_ss', 'inak.INaK', 'ikb.IKb', 'inab.INab', 'icab.ICab', 'ipca.IpCa']
    y0 = sim.pre_simulate(5000, sim_type=1)    
    d = sim.simulate(protocol.get_voltage_change_endpoints()[-1], log_times=None, extra_log=['membrane.i_ion'] + extra_log)
    
    times = d['engine.time']    
    
    command_voltages = [protocol.get_voltage_at_time(t) for t in times]    
    tr = trace.Trace(protocol,
                     cell_params=None,
                     t=times,
                     y=command_voltages,  # simulator.model.V,
                     command_voltages=command_voltages,
                     current_response_info=sim.current_response_info,
                     default_unit=None)   
    
    return tr



# def get_model_response_with_myokit( simulator, protocol, prestep=None):    
    
#     model, p, s = myokit.load( "../mmt-model-files/ohara-cipa-v1-2017_VC.mmt" )    
                      
#     simulator = simulator_myokit.Simulator(model, protocol, max_step=1.0, abs_tol=1e-8, rel_tol=1e-8, vhold=-80) # 1e-12, 1e-14 # 1e-08, 1e-10  # max_step=1, atol=1E-2, rtol=1E-4 # defalt: abs_tol=1e-06, rel_tol=0.0001    
#     # simulator.reset_simulation_with_new_protocol( protocol )
#     simulator.simulation.set_constant('cell.mode', 1)  
    
#     if prestep == None:
#         print("There is no pre-step simulation.")
#     elif prestep == 15000:        
#         y0 = [-8.69999996e+01,  6.94732336e+00,  6.94736848e+00,  1.44992431e+02,
#                1.44992434e+02,  5.48328391e-05,  5.40431668e-05,  1.25617506e+00,
#                1.25618638e+00,  8.12231733e-03,  6.62326077e-01,  6.62326077e-01,
#                6.62326077e-01,  4.14582271e-01,  6.62326077e-01,  2.27721811e-04,
#                4.79645030e-01,  2.87189165e-01,  1.07103663e-03,  9.99468797e-01,
#                9.99468797e-01,  5.45740810e-04,  9.99468797e-01,  9.99468797e-01,
#                2.96634937e-09,  9.99999988e-01,  9.99999988e-01,  9.99999988e-01,
#                9.99999988e-01,  9.99999988e-01,  4.78979614e-04,  9.99999988e-01,
#                9.99999988e-01,  9.28750206e-06,  9.23466020e-06,  1.96054631e-04,
#                2.15667189e-04,  9.97012407e-01,  1.27419629e-07,  1.59274616e-07,
#                2.47073549e-04]
#         simulator.set_initial_values(y0)      
#     else:
#         simulator.pre_simulate(pre_step=prestep, sim_type=1)
    
#     d = simulator.simulate(protocol.get_voltage_change_endpoints()[-1], log_times=None, extra_log=['ina.INa', 'inal.INaL', 'ito.Ito', 'ical.ICaL', 'ikr.IKr', 'iks.IKs', 'ik1.IK1'])     
#     times = d['engine.time']    
#     command_voltages = [protocol.get_voltage_at_time(t) for t in times]    
#     tr = trace.Trace(protocol,
#                      cell_params=None,
#                      t=times,
#                      y=command_voltages,  # simulator.model.V,
#                      command_voltages=command_voltages,
#                      current_response_info=simulator.current_response_info,
#                      default_unit=None)         
#     return tr