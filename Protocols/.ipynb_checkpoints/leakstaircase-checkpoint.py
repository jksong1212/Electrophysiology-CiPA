import os # import listdir
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bisect

def get_voltage_clamp_step_tangent_yIntercept(t1, V1, t2, V2):
    '''
    y=ax+b
    y-V1 = (V2-V1)/(t2-t1)*(x-t1)  <-  (t1, V1) , (t2, V2)
    return a, b
    '''
    a = (V2 - V1)/(t2 - t1)
    b = -a*t1 + V1
    return a, b

class LeakStaircase(): # suggested by 
    def __init__(self, return_capmask=False):
        # My 'test6_v3'/staircase-ramp protocol
        # model: myokit model
        # return_capmask: if True, return an extra function that takes time series
        #                 as argument and return a mask to filter off capacitance
        #                 effect.

        self.type = 'VC'
        
        tpre  = 0.2           # Time before step to variable V
        tstep = 0.5           # Time at variable V
        tpost = 0.1           # Time after step to variable V
        vhold = -80e-3
        vmin = -60e-3#-100e-3
        vmax = 40e-3
        vres = 20e-3        # Difference in V between steps
        v = np.arange(vmin, vmax + vres, vres)

        self.steps = []        # (V ,duration, 'step')  (V_start, V_end, duration, 'ramp')
        # Leak estimate
        self.steps += [(0.25, vhold,  'step')]
        self.steps += [(0.05, -120e-3, 'step')]
        self.steps += [(400e-3, -30e-3, 'ramp')]  # ramp step
        # Staircase
        self.steps += [(0.2, vhold, 'step')]
        self.steps += [(1.0, 40e-3, 'step')]
        self.steps += [(0.5, -120e-3, 'step')]
        self.steps += [(1.0, vhold, 'step')]
        for vstep in v[1::]:
            self.steps += [(tstep, vstep, 'step')]
            self.steps += [(tstep, vstep-vres, 'step')]
        for vstep in v[::-1][:-1]:
            self.steps += [(tstep, vstep, 'step')]
            self.steps += [(tstep, vstep-2*vres, 'step')]
        self.steps += [(1.0 - tstep, vhold, 'step')]  # extend a bit the ending...
        # EK estimate
        self.steps += [(tstep, 40e-3, 'step')]
        self.steps += [(10e-3, -70e-3, 'step')]  # Michael's suggestion
        self.steps += [(tstep-10e-3, -120e-3, 'ramp')]  # second ramp step
        self.steps += [(100, vhold, 'step')]
        
        self.voltage_change_startpoints = [0]
        cumulative_time = 0
        for step in self.steps[0:-1]:            
            cumulative_time += step[0]
            self.voltage_change_startpoints.append(cumulative_time)

        self.voltage_change_endpoints = []
        cumulative_time = 0
        for step in self.steps:
            cumulative_time += step[0]
            self.voltage_change_endpoints.append(cumulative_time)
            
#         print(self.voltage_change_startpoints)
#         print(self.voltage_change_endpoints)

    def voltage_at_time(self, time):        
        if time >= 0.300 and time < 0.700001:
            return  -150e-3 + 0.1 * time
        elif time >= 14.410 and time < 14.510001 :
            return 5.694 - 0.4 * time
        else:
            step_index = bisect.bisect_left( self.voltage_change_endpoints, time )            
            return self.steps[step_index][1] 
        
      
    
    
    

#     # Set ramp bit
#     model.get('membrane.V').set_rhs(
#                 'piecewise('
#                 +
#                 'engine.time >= 0.300 and engine.time < 0.700001,'
#                 + '-150e-3 + 0.1 * engine.time'
#                 +
#                 ', engine.time >= 14.410 and engine.time < 14.510001,'
#                 + ' + 5.694 - 0.4 * engine.time'
#                 +
#                 ', engine.pace)')

#     if return_capmask:

#         def capmask(times, capmaskdt=capmaskdt):
#             fcap = np.ones(times.shape)
#             currentt = 0
#             for v, dur in steps:
#                 idxi = np.where(times > currentt)[0][0] - 1  # inclusive
#                 idxf = np.where(times > currentt + capmaskdt)[0][0]
#                 fcap[idxi:idxf] = 0
#                 currentt += dur
#             return fcap

#         return model, steps, capmask
#     else:
#         return model, steps


