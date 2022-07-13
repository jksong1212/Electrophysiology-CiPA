import pickle
import matplotlib.pyplot as plt

import mod_kernik as kernik

#############################################
from scipy.integrate import ode, solve_ivp
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bisect

sys.path.append('../')
from Protocols.pacing_protocol import PacingProtocol
from Protocols.leakstaircase import LeakStaircase

sys.path.append('../')
import scipy_simulator

sys.path.append('../models')
from Models.br1977 import BR1977

import mod_trace as trace
#############################################

def get_high_fitness(ga_result):
    best_individual = ga_result.generations[0][0]

    for i, gen in enumerate(ga_result.generations):
        best_in_gen = ga_result.get_high_fitness_individual(i)
        if best_in_gen.fitness > best_individual.fitness:
            best_individual = best_in_gen

    return best_individual

def br1977( protocol):
        model = BR1977(protocol)
        simulator = scipy_simulator.Simulator(model)
        model.name = "Beeler and Reuter 1977"      
        times = [0, protocol.get_voltage_change_endpoints()[-1]]             
        simulator.simulate(times)                 
                
        model.current_response_info = trace.CurrentResponseInfo()
        if len(simulator.solver.y) < 200:
            list(map(model.differential_eq, simulator.solver.t, simulator.solver.y.transpose()))
        else:            
            list(map(model.differential_eq, simulator.solver.t, simulator.solver.y))               
        
        tr = trace.Trace(protocol,
                    cell_params=None,
                    t=model.times,
                    y=model.V,
                    command_voltages=model.V,            
                    current_response_info=model.current_response_info,
                    default_unit=None)
        
        return tr
        
def plot_current_conributions():
    trial_conditions = "trial_steps_ramps_Kernik_16_10_4_-120_60"
    currents = ['I_Na', 'I_si', 'I_K1', 'I_x1']

    for i, current in enumerate(currents):
        ga_result = pickle.load(open(f'ga_results/{trial_conditions}/ga_results_{current}_artefact_True', 'rb'))
        best_individual = get_high_fitness(ga_result)
        proto = best_individual.protocol
        
        tr = br1977(proto)
        # k = kernik.KernikModel(is_exp_artefact=True)        
        # tr = k.generate_response(proto, is_no_ion_selective=False)

        tr.plot_currents_contribution(current, is_shown=True, title=current,
                saved_to=f'./ga_results/{trial_conditions}/{current}.svg')

def main():
    plot_current_conributions()

if __name__ == '__main__':
    main()
