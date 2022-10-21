import copy
import numpy as np
import bisect


def get_modified_current( params, current ):  
    
    times = params['time']
    vc_protocol = params['vc']    
    neighbor_radius = params['neighbor_radius']
    
    method = params['method']
         
    pre_step = vc_protocol.steps[0]
    pre_step_index = 0
    modified_current = copy.copy(current)
    
       
    for i, t in enumerate(times):
        """Gets the voltage based on provided steps for the specified time."""
        # pre_V = vc_protocol.get_voltage_at_time(times[i-1])
        V = vc_protocol.get_voltage_at_time(t)
                        
        step_index = bisect.bisect_left( vc_protocol.get_voltage_change_endpoints(), t )
        current_step = vc_protocol.steps[step_index]
        if (pre_step_index != step_index) :
                        
            neighboring_value = current[i-10:i-1].mean()
                
            if method == 0:
                modified_current[i] = neighboring_value                    
                for j in range(1, neighbor_radius+1):
                    left_neighbor_index = i - j 
                    right_neighbor_index = i + j                 
                    modified_current[left_neighbor_index] = neighboring_value                                    
                    modified_current[right_neighbor_index] = neighboring_value                    
            elif method == 1:
                if V - vc_protocol.get_voltage_at_time(times[i-1]) > 0 :
                    for j in range( 2*neighbor_radius+1):
                        neighbor_index = i - neighbor_radius + j                     
                        if (current[neighbor_index] - neighboring_value) < -10:
                            modified_current[neighbor_index] = neighboring_value                                    
                elif V - vc_protocol.get_voltage_at_time(times[i-1]) < 0 :
                    for j in range( 2*neighbor_radius+1):
                        neighbor_index = i - neighbor_radius + j                     
                        if (current[neighbor_index] - neighboring_value) > 10:
                            modified_current[neighbor_index] = neighboring_value  

            pre_step_index = step_index
            pre_step = current_step
            
    return modified_current




'''
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