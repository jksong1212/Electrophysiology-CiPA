import os, sys, copy
import time, glob

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import myokit


sys.path.append('../Protocols')
sys.path.append('../Lib')
import protocol_lib
import mod_trace
import mod_protocols




def get_tangent_yIntercept(p1 : list, p2 : list) :
    '''
    Get the tangent and y-intercept of a line(y=ax+b) from points p1 and p2
    p1 = (x1, y1)
    p2 = (x2, y2)
    y=ax+b  <- a: tangent  |  b: y-intercept
    return a, b
    '''
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    a = float('inf')
    if dx != 0:  a = dy/dx
    b = -a*p1[0] + p1[1]
    return a, b



class Simulator:
    """
    
    """
    def __init__(self, model=None, protocol=None, max_step=None, abs_tol=1e-06, rel_tol=0.0001, vhold=-80):
        '''
        
        '''
        self.name = 'Myokit'
        # Get model
        # self.model, self.protocol, self._script = myokit.load(model_path)
        self.model = copy.copy(model)
        self.protocol = copy.copy(protocol)
        self.max_step = max_step
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol                               
        # basename = os.path.basename(model_path)        
        # self.name = os.path.splitext(basename)[0]                        
        # 1. Create pre-pacing protocol
        self.reset_pre_simulation(vhold)
        # Create simulation
        self.reset_simulation_with_new_protocol(protocol)

        self.pre_sim_state = False

    def reset_pre_simulation(self, vhold):
        p_const = myokit.pacing.constant(vhold)
        self.pre_simulation = myokit.Simulation( self.model, p_const )
        self.pre_init_state = self.pre_simulation.state()

        # self.pre_simulation.set_tolerance(abs_tol=self.abs_tol, rel_tol=self.rel_tol)  # 1e-12, 1e-14  # 1e-08 and rel_tol ¼ 1e-10
        # self.pre_simulation.set_max_step_size(self.max_step)

        # model = copy.copy(self.model)
        # p_const = protocol_lib.VoltageClampProtocol( [protocol_lib.VoltageClampStep(voltage=vhold, duration=100000)] )
        # if isinstance(p_const, protocol_lib.VoltageClampProtocol) or isinstance(p_const, mod_protocols.VoltageClampProtocol):
        #     model, p_const = self.transform_to_myokit_protocol(p_const, model)
        # self.pre_simulation = myokit.Simulation( model, p_const )        
        # self.pre_init_state = self.pre_simulation.state()

    def reset_simulation_with_new_protocol(self, protocol):
        model = copy.copy(self.model)        
        self.protocol = copy.copy(protocol)
        if isinstance(protocol, protocol_lib.VoltageClampProtocol) or isinstance(protocol, mod_protocols.VoltageClampProtocol):
            model, protocol = self.transform_to_myokit_protocol(protocol, model)
        self.simulation = myokit.Simulation(model, protocol)
        self.simulation.set_tolerance(abs_tol=self.abs_tol, rel_tol=self.rel_tol)  # 1e-12, 1e-14  # 1e-08 and rel_tol ¼ 1e-10
        self.simulation.set_max_step_size(self.max_step)
        # self.simulation.set_min_step_size(dtmin=0.000000000001)
        self.init_state = self.simulation.state()
        # self.simulation.set_protocol( protocol )


    def set_initial_values(self, y0):   
        '''        
        '''
        self.simulation.reset()     
        self.simulation.set_state(y0)
        self.pre_sim_state = True


    def set_simulation_params(self, parameters):
        '''
        parameters : dictionary
        '''            
        for key, value in parameters.items():        
            self.pre_simulation.set_constant(key, value)        
            self.simulation.set_constant(key, value)        
    


    def pre_simulate(self, pre_step, sim_type=0):   
        '''
        if pre_sim_type==0 : No pre simulation
        if pre_sim_type==1 : pre simulation
        if pre_sim_type==2 : constant pacing pre simulation
        '''
        self.simulation.reset()        
        
        if sim_type==0:            
            self.pre_simulation.reset()
            self.simulation.reset()
            self.pre_simulation.set_state(self.pre_init_state)
            self.simulation.set_state(self.pre_init_state)            
            self.simulation.pre(pre_step) # self.bcl*100
            
        elif sim_type==1:  # 
            self.pre_simulation.reset()
            self.simulation.reset()
            self.pre_simulation.set_state(self.pre_init_state)
            self.simulation.set_state(self.pre_init_state)            
            self.pre_simulation.pre(pre_step)
            self.simulation.set_state(self.pre_simulation.state())

        elif sim_type==2:  # 
            self.pre_simulation.reset()
            self.simulation.reset()
            self.pre_simulation.set_state(self.pre_init_state)
            self.simulation.set_state(self.pre_init_state)            
            self.pre_simulation.run(pre_step)
            self.simulation.set_state(self.pre_simulation.state())

        self.pre_sim_state = True
        
        return self.simulation.state()
        
    def simulate(self, end_time, log_times=None, extra_log=[], apd_variable=None, apd_threshold=None, log_interval=None):      
        
        if not self.pre_sim_state:
            self.simulation.reset()     
            self.simulation.set_state(self.init_state)
        
        # Run simulation               
        try:
            if apd_variable==None and apd_threshold==None:
                d = self.simulation.run(end_time,
                                        log_times = log_times,
                                        log = ['engine.time', 'membrane.V'] + extra_log,
                                        log_interval = log_interval
                                       )
                if extra_log:            
                    self.current_response_info = mod_trace.CurrentResponseInfo()
                    for i in range(len(d['engine.time'])):     
                        current_timestep = []
                        for name in extra_log:
                            current_timestep.append(mod_trace.Current(name=name.split('.')[1], value=d[name][i]))
                        self.current_response_info.currents.append(current_timestep)

                self.pre_sim_state = False
                
                for key, value in d.items():
                    d[key] = np.array(value)

                return d
            else:                
                d, apds = self.simulation.run(end_time,
                                              log_times = log_times,
                                              log = ['engine.time', 'membrane.V'] + extra_log,
                                              log_interval = log_interval,
                                              apd_variable=apd_variable, apd_threshold=apd_threshold
                                             )            
                if extra_log:            
                    self.current_response_info = mod_trace.CurrentResponseInfo()
                    for i in range(len(d['engine.time'])):     
                        current_timestep = []
                        for name in extra_log:
                            current_timestep.append(mod_trace.Current(name=name.split('.')[1], value=d[name][i]))
                        self.current_response_info.currents.append(current_timestep)

                self.pre_sim_state = False
                
                for key, value in d.items():
                    d[key] = np.array(value)

                return d, apds
            
        except myokit.SimulationError:
            return float('inf')
            
        


    def transform_to_mmt_ramp_script(self, ramp, time_start):            
        time_end = time_start + ramp.duration
        m, b = get_tangent_yIntercept((time_start, ramp.voltage_start), (time_end, ramp.voltage_end))        
        mmt_script = f'engine.time >= {time_start} and engine.time < {time_end}, {b} + {m} * engine.time, '                             
        return mmt_script


    def transform_to_myokit_protocol(self, VC_protocol, model_myokit=None):
        protocol_myokit = myokit.Protocol()
        ramp_script = 'piecewise('
        end_times = 0
        for step in VC_protocol.steps:            
            if isinstance(step, protocol_lib.VoltageClampStep) or isinstance(step, mod_protocols.VoltageClampStep) :
                protocol_myokit.add_step(step.voltage, step.duration)
            elif isinstance(step, protocol_lib.VoltageClampRamp)or isinstance(step, mod_protocols.VoltageClampRamp):
                protocol_myokit.add_step(0.5*(step.voltage_end+step.voltage_start), step.duration)
                ramp_script += self.transform_to_mmt_ramp_script(step, end_times) 
            end_times += step.duration
        
        step = VC_protocol.steps[-1]
        if isinstance(step, protocol_lib.VoltageClampStep) or isinstance(step, mod_protocols.VoltageClampStep) :
            protocol_myokit.add_step(step.voltage, 1)
        elif isinstance(step, protocol_lib.VoltageClampRamp)or isinstance(step, mod_protocols.VoltageClampRamp):
            protocol_myokit.add_step(step.voltage_end, 1)
            
        ramp_script += 'engine.pace)'        
        if len(ramp_script)>22:
            try:
                model_myokit.get('membrane.VC').set_rhs(ramp_script)
            except:
                model_myokit.get('membrane.V').set_rhs(ramp_script)
        return model_myokit, protocol_myokit
    
    

        
if __name__=='__main__':
    
    start_time = time.time()
   
   

    print("--- %s seconds ---"%(time.time()-start_time))
