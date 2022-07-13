import os # import listdir
from math import log, sqrt, floor
import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import bisect

class PacingProtocol(): # suggested by 
    def __init__(self, level=1, start=20, length=0.5, period=1000, multiplier=0, default_time_unit='ms'):
        
        self.type = 'AP'
        
        self._pace = 1
        self._stim_mag = 1
        self._level = level
        self._start = start
#         self._end = end
        self._length = length
        self._period = period
        self._multiplier = multiplier

        self._time_conversion = 1000.0
        if default_time_unit == 's':
            self._time_conversion = 1.0
            default_unit = 'standard'
        else:
            self._time_conversion = 1000.0
            default_unit = 'milli'


    def pacing(self, time):     
        # stim_amplitude = protocol.stim_amplitude * 1E-3 * self._time_conversion
        stim_start = self._start * 1E-3 * self._time_conversion
        stim_duration = self._length * 1E-3 * self._time_conversion
#         stim_end = self._end * 1E-3 * self._time_conversion
        i_stim_period = self._period * 1E-3 *self._time_conversion / self._pace

        if self._time_conversion == 1:
            denom = 1E9
        else:
            denom = 1

        pace = (1.0 if time - stim_start - i_stim_period*floor((time - stim_start)/i_stim_period) <= stim_duration \
#                 and time <= stim_end \
                and time >= stim_start else 0) / denom
  
        return pace

    