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