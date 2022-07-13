import os, sys
import time, glob

import random
import math
from math import log, sqrt, floor, exp
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from functools import partial 
from tqdm import tqdm
# import pickle
# import bisect

sys.path.append('../Protocols')
sys.path.append('../Lib')
import protocol_lib
import mod_trace
        


'''
The 2017 "CiPA v1" update [1] of the O'Hara et al. model of the human
ventricular AP [2].

This Myokit implementation was based on CellML code [3], published by Chang
et al. [4]. The authors checked the CellML output (after converting to
Chaste using PyCML) against derivatives calculated with the original code
published by the FDA [5].

The model differs from the original O'Hara model [2] in the following
aspects:
    - The IKr formulation was replaced entirely, as described in [1,4].
    - Conductances for INaL, ICaL, IKs, and IK1 were rescaled, as described
    in [6].

References:

[1] Li, Dutta et al., Colatsky (2017) Improving the In Silico Assessment o
    Proarrhythmia Risk by Combining hERG (Human Ether-à-go-go-Related Gene)
    Channel–Drug Binding Kinetics and Multichannel Pharmacology.
    Circulation: Arrhythmia and Electrophysiology.
    doi: 10.1161/CIRCEP.116.004628

[2] O'Hara, Virág, Varró, Rudy (2011) Simulation of the Undiseased Human
    Cardiac Ventricular Action Potential: Model Formulation and
    Experimental Validation. PLoS Computational Biology
    doi: 10.1371/journal.pcbi.1002061

[3] https://models.cellml.org/e/4e8/ohara_rudy_cipa_v1_2017.cellml/view

[4] Chang, Dutta et al., Li (2017) Uncertainty Quantification Reveals the
    Importance of Data Variability and Experimental Design Considerations
    for in Silico Proarrhythmia Risk Assessment. Frontiers in Physiology.
    doi: 10.3389/fphys.2017.00917

[5] https://github.com/FDA/CiPA/blob/master/AP_simulation/models/newordherg_qNet.c

[6] Dutta, Chang et al. Li (2017) Optimization of an In silico Cardiac Cell
    Model for Proarrhythmia Risk Assessment. Frontiers in Physiology.
    doi: 10.3389/fphys.2017.00616
'''


class Ishi():
    Mg_in = 1
    SPM_in = 0.005
    phi = 0.9

    def __init__(self):
        pass

    @classmethod
    def I_K1(cls, V, E_K, y1, K_out, g_K1):
        IK1_alpha = (0.17*exp(-0.07*((V-E_K) + 8*cls.Mg_in)))/(1+0.01*exp(0.12*(V-E_K)+8*cls.Mg_in))
        IK1_beta = (cls.SPM_in*280*exp(0.15*(V-E_K)+8*cls.Mg_in))/(1+0.01*exp(0.13*(V-E_K)+8*cls.Mg_in));
        Kd_spm_l = 0.04*exp(-(V-E_K)/9.1);
        Kd_mg = 0.45*exp(-(V-E_K)/20);
        fo = 1/(1 + (cls.Mg_in/Kd_mg));
        y2 = 1/(1 + cls.SPM_in/Kd_spm_l);
        
        d_y1 = (IK1_alpha*(1-y1) - IK1_beta*fo**3*y1);

        gK1 = 2.5*(K_out/5.4)**.4 * g_K1 
        I_K1 = gK1*(V-E_K)*(cls.phi*fo*y1 + (1-cls.phi)*y2);

        return [I_K1, y1]



class ExperimentalArtefactsThesis():
    """
    Experimental artefacts from Lei 2020
    For a cell model that includes experimental artefacts, you need to track
    three additional differential parameters: 

    The undetermined variables are: v_off, g_leak, e_leak
    Given the simplified model in section 4c,
    you can make assumptions that allow you to reduce the undetermined
    variables to only:
        v_off_dagger – mostly from liquid-junction potential
        g_leak_dagger
        e_leak_dagger (should be zero)
    """
    def __init__(self, g_leak=1, v_off=-2.8, e_leak=0, r_pipette=2E-3,
                 comp_rs=.8, r_access_star=20E-3,
                 c_m_star=60, tau_clamp=.8E-3, c_p_star=4, tau_z=7.5E-3,
                 tau_sum=40E-3, comp_predrs=None):
        """
        Parameters:
            Experimental measures:
                r_pipette – series resistance of the pipette
                c_m – capacitance of the membrane
            Clamp settings
                alpha – requested proportion of series resistance compensation
        """
        self.g_leak = g_leak
        self.e_leak = e_leak
        self.v_off = v_off
        self.c_p = c_p_star * .95
        self.c_p_star = c_p_star
        self.r_pipette = r_pipette
        self.c_m = c_m_star * .95
        self.r_access = r_access_star * .95
        self.comp_rs = comp_rs # Rs compensation
        self.r_access_star = r_access_star
        self.c_m_star = c_m_star
        self.tau_clamp = tau_clamp
        self.tau_z = tau_z
        self.tau_sum = tau_sum

        if comp_predrs is None:
            self.comp_predrs = comp_rs # Rs prediction







        
class Membrane():
    def __init__(self):        
        self.V = -8.80019046500000002e1 #
        self.y0 = [self.V]

    def d_V(self, Iions):
        '''
        in [mV]
        Membrane potential
        '''
        return [-Iions]
    
class Stimulus():
    def __init__(self):
        self.amplitude = -80 # [uA/cm^2]
        self.I = 0
        
    def cal_stimulation(self, pace=0):
        self.I = self.amplitude * pace     
        return self.I

class Phys():
    def __init__(self):
        '''        
        Physical constants
        Page 2 of the appendix to [2]
        '''
        self.R = 8314           # [J/kmol/K] : Gas constant
        self.T = 310            # [K] : Temperature
        self.F = 96485          # [C/mol] : Faraday's constant
        self.RTF  = self.R*self.T/self.F
        self.FRT  = self.F/(self.R*self.T)
        self.FFRT = self.F*self.FRT
        self.zna = 1
        self.zca = 2
        self.zk = 1        

class Cell():
    def __init__(self, phys):
        '''        
        # Cell geometry
        # Page 6 of the appendix to [2]      
        # The type of cell. Endo=0, Epi=1, Mid=2
        '''                        
        self.mode = -1  # The type of cell. Endo=0, Epi=1, Mid=2
        self.L = 0.01  # [cm] Cell length
        self.rad = 0.0011  # [cm] cell radius
        self.vcell = 1000 * 3.14 * self.rad * self.rad * self.L # [uL] Cell volume
        self.Ageo = 2*3.14 * self.rad * self.rad + 2*3.14 * self.rad * self.L   # [cm^2] Geometric cell area
        self.Acap = 2 * self.Ageo            # [cm^2] Capacitative membrane area
        self.vmyo = 0.68 * self.vcell        # [uL] Volume of the cytosolic compartment
        self.vnsr = 0.0552 * self.vcell      # [uL] Volume of the NSR compartment
        self.vjsr = 0.0048 * self.vcell      # [uL] Volume of the JSR compartment
        self.vss = 0.02 * self.vcell         # [uL] Volume of the Submembrane space near the T-tubules
        self.AF = self.Acap / phys.F * 1.0        # [uF/cm^2] in [uF*mol/C]

class Extra():
    '''
    Extracellular concentrations
    Page 5 of the appendix to [2]
    '''
    def __init__(self):        
        self.Nao = 140 # [mmol/L] : Extracellular Na+ concentration
        self.Cao = 1.8 # [mmol/L] : Extracellular Ca2+ concentration
        self.Ko  = 5.4 # [mmol/L] : Extracellular K+ concentration


class Nernst():
    '''
    Reversal potentials
    Page 6 of the appendix to [2]
    '''
    def __init__(self, phys, cell, extra):               
        self.phys = phys
        self.cell = cell
        self.extra = extra  
            
    def calculate(self, Nai, Ki):        
        self.ENa = self.phys.RTF * log(self.extra.Nao / Nai)      # in [mV]  desc: Reversal potential for Sodium currents
        self.EK = self.phys.RTF * log(self.extra.Ko / Ki)      # in [mV]  desc: Reversal potential for Potassium currents                
        PKNa = 0.01833          # desc: Permeability ratio K+ to Na+
        self.EKs = self.phys.RTF * log((self.extra.Ko + PKNa * self.extra.Nao) / (Ki + PKNa * Nai)) # desc: Reversal potential for IKs  in [mV]


class CaMK(): 
    def __init__(self):
        '''
        CaMKII signalling
        '''
        # initial value
        self.CaMKt = 1.25840446999999998e-2

        self.y0 = [self.CaMKt]
                
    def d_CaMKt(self, CaMKt, Ca_ss):
        '''          
        '''
        KmCaMK = 0.15
        aCaMK  = 0.05
        bCaMK  = 0.00068
        CaMKo  = 0.05
        KmCaM  = 0.0015
        CaMKb  = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / Ca_ss)
        CaMKa  = CaMKb + CaMKt
        d_CaMKt = aCaMK * CaMKb * CaMKa - bCaMK * CaMKt
        self.f = 1 / (1 + KmCaMK / CaMKa) # Fraction of phosphorylated channels        
        
        return [d_CaMKt]



class INa():
    '''
    # INa: Fast sodium current
    # Page 6 of the appendix to [2]
    #
    # The fast sodium current is modelled using a Hodgkin-Huxley type formulation
    # including activation (m), slow and fast components of inactivation (h) and
    # recovery from inactivation (j). The slow component of inactivation and
    # recovery from inactivation have an alternative formulation for CaMKII-
    # phosphorylated channels.
    '''
    def __init__(self, phys, cell, extra):       
        self.phys = phys
        self.cell = cell
        self.extra = extra   
        
        # initial values    
        self.m = 7.34412110199999992e-3
        self.hf = 6.98107191299999985e-1
        self.hs = 6.98089580099999996e-1
        self.j = 6.97990843200000044e-1
        self.hsp = 4.54948552499999992e-1
        self.jp = 6.97924586499999999e-1

        self.y0 = [self.m, self.hf, self.hs, self.j, self.hsp, self.jp]

        #: Maximum conductance of INa channels
        self.GNa = 75.0      
        self.G_adj = 1.0  
            
    def diff_eq(self, V, m, hf, hs, j, hsp, jp, camk, nernst):
        '''
        Activation gate for INa channels
        '''     
        # m-gates             
        sm  = 1.0 / (1.0 + exp(-(V + 39.57)/9.871))  # desc: Steady state value for m-gate         
        self.tm  = 1.0 / (6.765 * exp((V + 11.64) / 34.77) + 8.552 * exp(-(V + 77.42) / 5.955)) # desc: Time constant for m-gate   in [ms]                
        d_m = (sm - m) / self.tm           

        # h-gates        
        shift_INa_inact = 0.0

        sh = 1.0 / (1.0 + exp((V + 82.9-shift_INa_inact) / 6.086))   # desc: Steady-state value for h-gate
        thf = 1.0 / (1.432e-5 * exp(-(V + 1.196 - shift_INa_inact) / 6.285) + 6.1490 * exp((V + 0.5096 - shift_INa_inact) / 20.27)) # desc: Time constant for fast development of inactivation in INa   in [ms]
        ths = 1.0 / (0.009794 * exp(-(V + 17.95-shift_INa_inact) / 28.05) + 0.3343 * exp((V + 5.7300 - shift_INa_inact) / 56.66))  # desc: Time constant for slow development of inactivation in INa  in [ms]
        Ahf = 0.99 # : Fraction of INa channels with fast inactivation
        Ahs = 1.0 - Ahf # : Fraction of INa channels with slow inactivation        
        d_hf = (sh - hf) / thf   # desc: Fast componennt of the inactivation gate for INa channels
        d_hs = (sh - hs) / ths   # desc: Slow componennt of the inactivation gate for non-phosphorylated INa channels
        h = Ahf * hf + Ahs * hs   # desc: Inactivation gate for INa
        
        # j-gates
        sj = sh # desc: Steady-state value for j-gate in INa
        tj = 2.038 + 1 / (0.02136 * exp(-(V + 100.6 - shift_INa_inact) / 8.281) + 0.3052 * exp((V + 0.9941 - shift_INa_inact) / 38.45)) # desc: Time constant for j-gate in INa in [ms]        
        d_j = (sj - j) / tj # desc: Recovery from inactivation gate for non-phosphorylated INa channels


        # Phosphorylated channels
        thsp = 3 * ths # desc: Time constant for h-gate of phosphorylated INa channels  in [ms]        
        shsp = 1 / (1 + exp((V + 89.1 - shift_INa_inact) / 6.086)) # desc: Steady-state value for h-gate of phosphorylated INa channels               
        d_hsp = (shsp - hsp) / thsp # desc: Slow componennt of the inactivation gate for phosphorylated INa channels
        hp = Ahf * hf + Ahs * hsp # desc: Inactivation gate for phosphorylated INa channels
        tjp = 1.46 * tj # desc: Time constant for the j-gate of phosphorylated INa channels     in [ms]        
        d_jp = (sj - jp) / tjp # desc: Recovery from inactivation gate for phosphorylated INa channels
        
        # Current                
        INa = self.G_adj * self.GNa * (V - nernst.ENa) * m**3 * ((1.0 - camk.f) * h * j + camk.f * hp * jp) # in [uA/uF]  desc: Fast sodium current
    
        return [d_m, d_hf, d_hs, d_j, d_hsp, d_jp], INa



class INaL():
    '''
    # INaL: Late component of the sodium current
    # Page 7 of the appendix to [2]
    '''
    def __init__(self, phys, cell, extra):        
        self.phys = phys
        self.cell = cell
        self.extra = extra  

        # initial values            
        self.mL         = 1.88261727299999989e-4
        self.hL         = 5.00854885500000013e-1
        self.hLp        = 2.69306535700000016e-1

        self.y0 = [self.mL, self.hL, self.hLp]

        #: Maximum conductance       
        self.GNaL = 0.0075    # 'Endocardial' : 0.0075 ,   'Epicardial' : 0.0075 * 0.6,   'Mid-myocardial' : 0.0075 ,
        self.G_adj = 1.0
            
    def diff_eq(self, V, mL, hL, hLp, camk, nernst, ina):        

        sm = 1 / (1 + exp((V + 42.85 ) / -5.264 ))  #desc: Steady state value of m-gate for INaL
        d_mL = (sm - mL) / ina.tm # desc: Activation gate for INaL

        th = 200  # [ms] : Time constant for inactivation of non-phosphorylated INaL  in [ms]
        sh = 1 / (1 + exp((V + 87.61 ) / 7.488 ))  # desc: Steady-state value for inactivation of non-phosphorylated INaL        
        d_hL = (sh - hL) / th # desc: Inactivation gate for non-phosphorylated INaL channels

        thp = 3 * th           # in [ms]     desc: Time constant for inactivation of phosphorylated INaL
        shp = 1 / (1 + exp((V + 93.81 ) / 7.488 ))   #  desc: Steady state value for inactivation of phosphorylated INaL
        d_hLp = (shp - hLp) / thp   # desc: Inactivation gate for phosphorylated INaL
                    
        # Current                
        INaL = self.G_adj * self.GNaL * (V - nernst.ENa) * mL * ((1 - camk.f) * hL + camk.f * hLp)
    
        return [d_mL, d_hL, d_hLp], INaL


class Ito():
    '''
    Ito :: Transient outward Potassium current
    page 8
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    
        
        # initial values            
        self.a    = 1.00109768699999991e-3
        self.iF   = 9.99554174499999948e-1
        self.iS   = 5.86506173600000014e-1
        self.ap   = 5.10086293400000023e-4
        self.iFp  = 9.99554182300000038e-1
        self.iSp  = 6.39339948199999952e-1

        self.y0 = [self.a, self.iF, self.iS, self.ap, self.iFp, self.iSp]
        
        self.delta_epi = 1.0
        
        #: Maximum conductance
        self.Gto = 0.02    # 'Endocardial' : 0.02 ,   'Epicardial' : 4.0*0.02   'Mid-myocardial' : 4.0*0.02,
        self.G_adj = 1.0

         
            
    def diff_eq(self, V, a, iF, iS, ap, iFp, iSp, camk, nernst):
                
        one = 1.0 / (1.2089 * (1 + exp(-(V - 18.4099) / 29.3814)))
        two = 3.5 / (1 + exp((V + 100) / 29.3814))
        ta = 1.0515 / (one + two)  # desc: Time constant for Ito activation  in [ms]        
        ass = 1.0 / (1.0 + exp(-(V - 14.34) / 14.82))  # desc: Steady-state value for Ito activation
        d_a = (ass - a) / ta   # desc: Ito activation gate
        
        iss = 1.0 / (1.0 + exp((V + 43.94) / 5.711))   # desc: Steady-state value for Ito inactivation
        
        self.delta_epi = 1.0
        if self.cell.mode==1:  
            self.delta_epi = 1.0 - (0.95 / (1 + exp((V + 70.0) / 5.0)))   # desc: Adjustment for different cell types

        tiF_b = (4.562 + 1 / (0.3933 * exp(-(V+100) / 100) + 0.08004 * exp((V + 50) / 16.59)))  # desc: Time constant for fast component of Ito inactivation     in [ms]
        tiS_b = (23.62 + 1 / (0.001416 * exp(-(V + 96.52) / 59.05) + 1.780e-8 * exp((V + 114.1) / 8.079)))   # desc: Time constant for slow component of Ito inactivation  in [ms]
        tiF = tiF_b * self.delta_epi
        tiS = tiS_b * self.delta_epi
        AiF = 1.0 / (1.0 + exp((V - 213.6) / 151.2))  # desc: Fraction of fast inactivating Ito channels
        AiS = 1.0 - AiF        
        d_iF = (iss - iF) / tiF  # desc: Fast component of Ito activation        
        d_iS = (iss - iS) / tiS  # desc: Slow component of Ito activation
        
        i = AiF * iF + AiS * iS     # desc: Inactivation gate for non-phosphorylated Ito
        assp=1.0/(1.0+exp(-(V-24.34)/14.82))        
        d_ap = (assp - ap) / ta
            
        dti_develop = 1.354 + 1e-4 / (exp((V - 167.4) / 15.89) + exp(-(V - 12.23) / 0.2154))
        dti_recover = 1 - 0.5 / (1 + exp((V+70) / 20))
        tiFp = dti_develop * dti_recover * tiF # desc: Time constant for fast component of inactivation of phosphorylated Ito channels   in [ms]
        tiSp = dti_develop * dti_recover * tiS  # desc: Time constant for slot component of inactivation of phosphorylated Ito channels  in [ms]        
        d_iFp = (iss - iFp) / tiFp # desc: Fast component of inactivation of phosphorylated Ito channels        
        d_iSp = (iss - iSp) / tiSp # desc: Slow component of inactivation of phosphorylated Ito channels
        
        ip = AiF * iFp + AiS * iSp  # desc: Inactivation gate for phosphorylated Ito channels
        
        # Current        
        Ito = self.Gto*self.G_adj * (V - nernst.EK) * ((1 - camk.f) * a * i + camk.f * ap * ip) # desc: Transient outward Potassium current
    
        return [d_a, d_iF, d_iS, d_ap, d_iFp, d_iSp], Ito


class ICaL():
    '''
    ICaL  :: L-type Calcium current
    ICaNa :: Sodium current through the L-type Calcium channel
    ICaK  :: Potassium current through the L-type Calcium channel
    Page 9
    The ICaL channel is modeled using activation, inactivation (fast and slow),
    Ca-dependent inactivation (fast and slow) and recovery from Ca-dependent
    inactivation.
    Inactivation and Ca-dependent inactivation have an alternative formulation
    for CaMKII phosphorylated channels.
    '''
    def __init__(self, phys, cell, extra):    
        self.phys = phys
        self.cell = cell
        self.extra = extra      
        
        # initial values            
        self.d            =  2.34e-9
        self.ff           =  9.99999990900000024e-1
        self.fs           =  9.10241277699999962e-1
        self.fcaf         =  9.99999990900000024e-1
        self.fcas         =  9.99804677700000033e-1
        self.jca          =  9.99973831200000052e-1
        self.nca          =  2.74941404400000020e-3
        self.ffp          =  9.99999990900000024e-1
        self.fcafp        =  9.99999990900000024e-1
        
        self.y0 = [self.d, self.ff, self.fs, self.fcaf, self.fcas, self.jca, self.nca, self.ffp, self.fcafp, ]

        #: Maximum conductance
        self.PCa = 0.0001   # 'Endocardial' : PCa_b(0.0001) ,   'Epicardial' : PCa_b*1.2  'Mid-myocardial' : PCa_b*2.5
        self.G_adj = 1.0
                      
    def diff_eq(self, V, 
                d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp,
                cass, nass, kss,
                camk, nernst):
        
        vfrt = V * self.phys.FRT
        vffrt = V * self.phys.FFRT
        
        # Activation
        dss = 1.0 / (1.0 + exp(-(V + 3.94) / 4.23)) # Steady-state value for activation gate of ICaL channel
        td = 0.6 + 1.0 / (exp(-0.05 * (V + 6)) + exp(0.09 * (V + 14)))  # Time constant for activation gate of ICaL channel   in [ms]
        d_d = (dss - d) / td   # Activation gate of ICaL channel
        
        # Inactivation
        fss = 1.0 / (1.0 + exp((V + 19.58) / 3.696)) # Steady-state value for inactivation gate of ICaL channel
        tff = 7.0 + 1.0 / (0.0045 * exp(-(V + 20) / 10) + 0.0045 * exp((V + 20) / 10))  # Time constant for fast inactivation of ICaL channels  in [ms]
        tfs = 1000 + 1.0 / (0.000035 * exp(-(V + 5) / 4) + 0.000035 * exp((V + 5) / 6)) # Time constant for fast inactivation of ICaL channels  in [ms]
        Aff = 0.6  # Fraction of ICaL channels with fast inactivation
        Afs = 1.0 - Aff # Fraction of ICaL channels with slow inactivation
        d_ff = (fss - ff) / tff   # Fast inactivation of ICaL channels
        d_fs = (fss - fs) / tfs   # Slow inactivation of ICaL channels        
        f = Aff * ff + Afs * fs  # Inactivation of ICaL channels
        
        # Ca-dependent inactivation
        fcass = fss  # Steady-state value for Ca-dependent inactivation of ICaL channels
        tfcaf = 7.0 + 1.0 / (0.04 * exp(-(V - 4.0) / 7.0) + 0.04 * exp((V - 4.0) / 7.0))  # Time constant for fast Ca-dependent inactivation of ICaL channels in [ms]
        tfcas = 100.0 + 1 / (0.00012 * exp(-V / 3) + 0.00012 * exp(V / 7))  # Time constant for slow Ca-dependent inactivation of ICaL channels  in [ms]
        Afcaf = 0.3 + 0.6 / (1 + exp((V - 10) / 10))  # Fraction of ICaL channels with fast Ca-dependent inactivation
        Afcas = 1.0 - Afcaf    # Fraction of ICaL channels with slow Ca-dependent inactivation
        d_fcaf = (fcass - fcaf) / tfcaf   # Fast Ca-dependent inactivation of ICaL channels
        d_fcas = (fcass - fcas) / tfcas   # Slow Ca-dependent inactivation of ICaL channels
        fca = Afcaf * fcaf + Afcas * fcas    # Ca-dependent inactivation of ICaL channels
        
        # Recovery from Ca-dependent inactivation
        tjca = 75.0  # [ms] : Time constant of recovery from Ca-dependent inactivation
        d_jca = (fcass - jca) / tjca  # Recovery from Ca-dependent inactivation
        
        # Inactivation of phosphorylated channels
        tffp = 2.5 * tff   # in [ms]  desc: Time constant for fast inactivation of phosphorylated ICaL channels
        d_ffp = (fss - ffp) / tffp   # Fast inactivation of phosphorylated ICaL channels
        fp = Aff * ffp + Afs * fs    # Inactivation of phosphorylated ICaL channels
        
        # Ca-dependent inactivation of phosphorylated channels
        tfcafp = 2.5 * tfcaf   # in [ms]  desc: Time constant for fast Ca-dependent inactivation of phosphorylated ICaL channels
        d_fcafp = (fcass - fcafp) / tfcafp  # Fast Ca-dependent inactivation of phosphorylated ICaL channels
        fcap = Afcaf * fcafp + Afcas * fcas   # Ca-dependent inactivation of phosphorylated ICaL channels
                
        Kmn = 0.002
        k2n = 1000
        km2n = jca * 1.0
        anca = 1.0 / (k2n / km2n + (1 + Kmn / cass)**4.0)  # Fraction of channels in Ca-depdent inactivation mode
        d_nca = anca * k2n - nca*km2n   # Fraction of channels in Ca-depdent inactivation mode
        
        # Total currents through ICaL channel        
        PhiCaL = float('inf')
        PhiCaNa = float('inf')
        PhiCaK = float('inf')
        v0 = 0
        B_1 = 2*self.phys.FRT
        A_1 = 4 * self.phys.FFRT * ( cass * exp(2 * vfrt) - 0.341 * self.extra.Cao) / B_1
        U_1 = B_1 * ( V - v0 )        
        if -1e-7 <= U_1 and U_1 <= 1e-7:
            PhiCaL = A_1 * (1.0-0.5*U_1)
        else :
            PhiCaL = (A_1 * U_1) / (exp(U_1)-1)
                
        B_2 = self.phys.FRT
        A_2 = 0.75 * self.phys.FFRT * ( nass * exp(vfrt) - self.extra.Nao) / B_2
        U_2 = B_2 * ( V - v0 )        
        if -1e-7 <= U_2 and U_2 <= 1e-7:
            PhiCaNa = A_2 * (1.0-0.5*U_2)
        else :
            PhiCaNa = (A_2 * U_2) / (exp(U_2)-1)
        
        B_3 = self.phys.FRT
        A_3 = 0.75 * self.phys.FFRT * ( kss * exp(vfrt) - self.extra.Ko) / B_3
        U_3 = B_3 * ( V - v0 )        
        if -1e-7 <= U_3 and U_3 <= 1e-7:
            PhiCaK = A_3 * (1.0-0.5*U_3)
        else :
            PhiCaK = (A_3 * U_3) / (exp(U_3)-1)


        # if vfrt==0:
        #     PhiCaL = 1 * 4 * (cass - 0.341 * self.extra.Cao)
        # else:
        #     PhiCaL = 4 * vffrt * (cass  * exp(2 * vfrt) - 0.341 * self.extra.Cao) / (exp(2 * vfrt) - 1)

        # if vfrt==0:
        #     PhiCaNa = 1 * 1 * (0.75 * nass - 0.75 * self.extra.Nao)
        # else:
        #     PhiCaNa = 1 * vffrt * (0.75 * nass  * exp(1 * vfrt) - 0.75  * self.extra.Nao) / (exp(1 * vfrt) - 1)

        # if vfrt==0:
        #     PhiCaK = 1 * 1 * (0.75 * kss - 0.75 * self.extra.Ko)
        # else:
        #     PhiCaK = 1 * vffrt * (0.75 * kss * exp(1 * vfrt) - 0.75  * self.extra.Ko ) / (exp(1 * vfrt) - 1)

        
        PCa = self.PCa*self.G_adj
        PCap   = 1.1      * PCa
        PCaNa  = 0.00125  * PCa
        PCaK   = 3.574e-4 * PCa
        PCaNap = 0.00125  * PCap
        PCaKp  = 3.574e-4 * PCap
        g  = d * (f  * (1.0 - nca) + jca * fca  * nca)   # Conductivity of non-phosphorylated ICaL channels
        gp = d * (fp * (1.0 - nca) + jca * fcap * nca)   # Conductivity of phosphorylated ICaL channels        
        ICaL   = (1.0 - camk.f) * PCa   * PhiCaL  * g + camk.f * PCap   * PhiCaL  * gp  # L-type Calcium current   in [uA/uF]
        ICaNa  = (1.0 - camk.f) * PCaNa * PhiCaNa * g + camk.f * PCaNap * PhiCaNa * gp   # Sodium current through ICaL channels  in [uA/uF]
        ICaK   = (1.0 - camk.f) * PCaK  * PhiCaK  * g + camk.f * PCaKp  * PhiCaK  * gp   # Potassium current through ICaL channels  in [uA/uF]
    
        return [d_d, d_ff, d_fs, d_fcaf, d_fcas, d_jca, d_nca, d_ffp, d_fcafp,], ICaL, ICaNa, ICaK


        
class IKr():
    '''
    # IKr: Rapid delayed rectifier potassium current
    # Described in [1,4].
    '''
    def __init__(self, phys, cell, extra):   
        self.phys = phys
        self.cell = cell
        self.extra = extra       
        
        # initial values       
        self.IC1           =  0.999637
        self.IC2           =  6.83207999999999982e-5
        self.C1            =  1.80144999999999990e-8
        self.C2            =  8.26618999999999954e-5
        self.O             =  1.55510000000000007e-4
        self.IO            =  5.67622999999999969e-5
        self.IObound       =  0
        self.Obound        =  0
        self.Cbound        =  0
        self.D             =  0             
        
        self.y0 = [self.IC1, self.IC2, self.C1, self.C2, self.O, self.IO, self.IObound, self.Obound, self.Cbound, self.D ]

        #: Maximum conductance
        self.GKr = 4.65854545454545618e-2 # [mS/uF]  in [mS/uF]
        self.G_adj = 1.0
                     
    def diff_eq(self, V, IC1, IC2, C1, C2, O, IO, IObound, Obound, Cbound, D, camk, nernst):
        
        A1 = 0.0264 # [mS/uF] in [mS/uF]
        A11 = 0.0007868 # [mS/uF] in [mS/uF]
        A2 = 4.986e-6 # [mS/uF]  in [mS/uF]
        A21 = 5.455e-6 # [mS/uF] in [mS/uF]
        A3 = 0.001214 # [mS/uF] in [mS/uF]
        A31 = 0.005509 # [mS/uF] in [mS/uF]
        A4 = 1.854e-5 # [mS/uF] in [mS/uF]
        A41 = 0.001416 # [mS/uF] in [mS/uF]
        A51 = 0.4492 # [mS/uF] in [mS/uF]
        A52 = 0.3181 #[mS/uF] in [mS/uF]
        A53 = 0.149 #[mS/uF] in [mS/uF]
        A61 = 0.01241 #[mS/uF] in [mS/uF]
        A62 = 0.3226 #[mS/uF] in [mS/uF]
        A63 = 0.008978 #[mS/uF] in [mS/uF]
        B1 = 4.631e-5 #[1/mV] in [1/mV]
        B11 = 1.535e-8 #[1/mV] in [1/mV]
        B2 = -0.004226 #[1/mV] in [1/mV]
        B21 = -0.1688 #[1/mV] in [1/mV]
        B3 = 0.008516 #[1/mV] in [1/mV]
        B31 = 7.771e-9 #[1/mV] in [1/mV]
        B4 = -0.04641 #[1/mV] in [1/mV]
        B41 = -0.02877 #[1/mV] in [1/mV]
        B51 = 0.008595 #[1/mV] in [1/mV]
        B52 = 3.613e-8 #[1/mV] in [1/mV]
        B53 = 0.004668 #[1/mV] in [1/mV]
        B61 = 0.1725 #[1/mV] in [1/mV]
        B62 = -6.57499999999999990e-4 #[1/mV] in [1/mV]
        B63 = -0.02215 #[1/mV] in [1/mV]
        q1 = 4.843
        q11 = 4.942
        q2 = 4.23
        q21 = 4.156
        q3 = 4.962
        q31 = 4.22
        q4 = 3.769
        q41 = 1.459
        q51 = 5
        q52 = 4.663
        q53 = 2.412
        q61 = 5.568
        q62 = 5
        q63 = 5.682
        Kt = 0 #[mS/uF] in [mS/uF]
        Ku = 0 #[mS/uF] in [mS/uF]
        Temp = 37
        Vhalf = 1 #[mV] in [mV]
        halfmax = 1
        n = 1
        Kmax = 0
        if -1e-11<D and D<1e-11:         
            D = 1e-11
        d_IC1 = -(A11 * exp(B11 * V) * IC1 * exp((Temp - 20) * log(q11) / 10) - A21 * exp(B21 * V) * IC2 * exp((Temp - 20) * log(q21) / 10)) + A51 * exp(B51 * V) * C1 * exp((Temp - 20) * log(q51) / 10) - A61 * exp(B61 * V) * IC1 * exp((Temp - 20) * log(q61) / 10)
        d_IC2 = A11 * exp(B11 * V) * IC1 * exp((Temp - 20) * log(q11) / 10) - A21 * exp(B21 * V) * IC2 * exp((Temp - 20) * log(q21) / 10) - (A3 * exp(B3 * V) * IC2 * exp((Temp - 20) * log(q3) / 10) - A4 * exp(B4 * V) * IO * exp((Temp - 20) * log(q4) / 10)) + A52 * exp(B52 * V) * C2 * exp((Temp - 20) * log(q52) / 10) - A62 * exp(B62 * V) * IC2 * exp((Temp - 20) * log(q62) / 10)
        d_C1 = -(A1 * exp(B1 * V) * C1 * exp((Temp - 20) * log(q1) / 10) - A2 * exp(B2 * V) * C2 * exp((Temp - 20) * log(q2) / 10)) - (A51 * exp(B51 * V) * C1 * exp((Temp - 20) * log(q51) / 10) - A61 * exp(B61 * V) * IC1 * exp((Temp - 20) * log(q61) / 10))
        d_C2 = A1 * exp(B1 * V) * C1 * exp((Temp - 20) * log(q1) / 10) - A2 * exp(B2 * V) * C2 * exp((Temp - 20) * log(q2) / 10) - (A31 * exp(B31 * V) * C2 * exp((Temp - 20) * log(q31) / 10) - A41 * exp(B41 * V) * O * exp((Temp - 20) * log(q41) / 10)) - (A52 * exp(B52 * V) * C2 * exp((Temp - 20) * log(q52) / 10) - A62 * exp(B62 * V) * IC2 * exp((Temp - 20) * log(q62) / 10))
        d_O = A31 * exp(B31 * V) * C2 * exp((Temp - 20) * log(q31) / 10) - A41 * exp(B41 * V) * O * exp((Temp - 20) * log(q41) / 10) - (A53 * exp(B53 * V) * O * exp((Temp - 20) * log(q53) / 10) - A63 * exp(B63 * V) * IO * exp((Temp - 20) * log(q63) / 10)) - (Kmax * Ku * exp(n * log(D)) / (exp(n * log(D)) + halfmax) * O - Ku * Obound)
        d_IO = A3 * exp(B3 * V) * IC2 * exp((Temp - 20) * log(q3) / 10) - A4 * exp(B4 * V) * IO * exp((Temp - 20) * log(q4) / 10) + A53 * exp(B53 * V) * O * exp((Temp - 20) * log(q53) / 10) - A63 * exp(B63 * V) * IO * exp((Temp - 20) * log(q63) / 10) - (Kmax * Ku * exp(n * log(D)) / (exp(n * log(D)) + halfmax) * IO - Ku * A53 * exp(B53 * V) * exp((Temp - 20) * log(q53) / 10) / (A63 * exp(B63 * V) * exp((Temp - 20) * log(q63) / 10)) * IObound)
        d_IObound = Kmax * Ku * exp(n * log(D)) / (exp(n * log(D)) + halfmax) * IO - Ku * A53 * exp(B53 * V) * exp((Temp - 20) * log(q53) / 10) / (A63 * exp(B63 * V) * exp((Temp - 20) * log(q63) / 10)) * IObound + Kt / (1 + exp(-(V - Vhalf) / 6.789 )) * Cbound - Kt * IObound
        d_Obound = Kmax * Ku * exp(n * log(D)) / (exp(n * log(D)) + halfmax) * O - Ku * Obound + Kt / (1 + exp(-(V - Vhalf) / 6.789 )) * Cbound - Kt * Obound
        d_Cbound = -(Kt / (1 + exp(-(V - Vhalf) / 6.789 )) * Cbound - Kt * Obound) - (Kt / (1 + exp(-(V - Vhalf) / 6.789 )) * Cbound - Kt * IObound)
        d_D = 0 # [1/ms]

        IKr = self.G_adj * self.GKr * sqrt(self.extra.Ko / 5.4 ) * O * (V - nernst.EK)  # in [A/F]
    
        return [ d_IC1, d_IC2, d_C1, d_C2, d_O, d_IO, d_IObound, d_Obound, d_Cbound, d_D,], IKr

        
    
class IKs():
    '''
    # IKs: Slow delayed rectifier potassium current
    # Page 11 of the appendix to [2]
    #
    # Modelled with two activation gates
    '''
    def __init__(self, phys, cell, extra):        
        self.phys = phys
        self.cell = cell
        self.extra = extra  
        
        # initial values            
        self.xs1 = 2.70775802499999996e-1
        self.xs2 = 1.92850342599999990e-4

        self.y0 = [self.xs1, self.xs2]

        #: Maximum conductance
        self.GKs = 0.0034   # 'Endocardial' : GKs_b = 0.0034 ,   'Epicardial' : GKs_b * 1.4  'Mid-myocardial' : GKs_b
        self.G_adj = 1.0
                    
    def diff_eq(self, V, xs1, xs2, Cai, camk, nernst):
        
        xs1ss  = 1.0 / (1.0 + exp(-(V + 11.60) / 8.932)) # desc: Steady-state value for activation of IKs channels
        txs1_max = 817.3
        txs1 = txs1_max + 1.0 / (2.326e-4 * exp((V + 48.28) / 17.80) + 0.001292 * exp(-(V + 210) / 230)) # desc: Time constant for slow, low voltage IKs activation
                        
        d_xs1 = (xs1ss - xs1) / txs1  # desc: Slow, low voltage IKs activation
        
        xs2ss = xs1ss
        txs2 = 1.0 / (0.01 * exp((V - 50) / 20) + 0.0193 * exp(-(V + 66.54) / 31.0)) # desc: Time constant for fast, high voltage IKs activation
        
        d_xs2 = (xs2ss - xs2) / txs2   # desc: Fast, high voltage IKs activation
        
        KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / Cai)**1.4) # desc: Maximum conductance for IKs
          
        IKs = self.GKs*self.G_adj * KsCa * xs1 * xs2 * (V - nernst.EKs)  # Slow delayed rectifier Potassium current
            
        return [d_xs1, d_xs2], IKs
    
class IK1():
    '''
    # IK1: Inward rectifier potassium current
    # Page 12 of the appendix to [2]
    #
    # Modelled with an activation channel and an instantaneous inactivation channel
    '''
    def __init__(self, phys, cell, extra):    
        self.phys = phys
        self.cell = cell
        self.extra = extra      
        
        # initial values            
        self.xk1 = 9.96759759399999945e-1
        
        self.y0 = [self.xk1]

        #: Maximum conductance
        self.GK1 = 0.1908*1.698   # 'Endocardial' : GK1_b = 0.1908  ,   'Epicardial' : GK1_b * 1.2 'Mid-myocardial' : GK1_b * 1.3  
        self.G_adj = 1.0
                    
    def diff_eq(self, V, xk1, camk, nernst):
        
        xk1ss = 1 / (1 + exp(-(V + 2.5538 * self.extra.Ko + 144.59) / (1.5692 * self.extra.Ko + 3.8115))) # Steady-state value for activation of IK1 channels  : sx in 2011
        txk1 = 122.2 / (exp(-(V + 127.2) / 20.36) + exp((V + 236.8) / 69.33))  # Time constant for activation of IK1 channels  : tx in 2011        
        d_xk1 = (xk1ss - xk1) / txk1  # Activation of IK1 channels        
        rk1 = 1.0 / (1.0 + exp((V + 105.8 - 2.6 * self.extra.Ko) / 9.493))   # Inactivation of IK1 channels    : r in 2011            

        IK1 = self.GK1*self.G_adj * sqrt(self.extra.Ko) * rk1 * xk1 * (V - nernst.EK)  # Inward rectifier Potassium current
            
        return [d_xk1], IK1


class IFunny():
    '''
    '''
    def __init__(self, phys, cell, extra):    
        self.phys = phys
        self.cell = cell
        self.extra = extra      
        
        # initial values            
        self.Xf = 6.403385049126155e-03
        
        self.y0 = [self.Xf]

        #: Maximum conductance
        self.G_F = 0.0435 #[nS/pF]  # 'Endocardial' : GK1_b = 0.1908  ,   'Epicardial' : GK1_b * 1.2 'Mid-myocardial' : GK1_b * 1.3  
        self.G_adj = 1.0
                    
    def diff_eq(self, V, Xf, camk, nernst):
                
        xF1 = 5.7897e-7 #[1/ms]            
        xF2 = -14.5897121702 #[mV]            
        xF5 = 20086.6502378844
        xF6 = 10.20235284528 #[mV]            
        xF_const = 23.94529134653 #[ms]            
        xF3 = xF5 * xF1     #  in [1/ms]
        xF4 = 1 / (1 / xF2 + 1 / xF6)  #  in [mV]
        a = xF1 * exp(V / xF2) #    in [1/ms]
        b = xF3 * exp(V / xF4) #    in [1/ms]
        inf = a / (a + b)
        tau = 1 / (a + b) + xF_const   #  in [ms]
        d_Xf = (inf - Xf) / tau  # desc: inactivation in i_f            
        NatoK_ratio = .491 # desc: Verkerk et al. 2013
        Na_frac = NatoK_ratio / (NatoK_ratio + 1)
        i_fNa = Na_frac * self.G_F*self.G_adj * Xf * (V - nernst.ENa) # in [A/F]
        i_fK = (1 - Na_frac) * self.G_F*self.G_adj * Xf * (V - nernst.EK) # in [A/F]
        i_f = i_fNa + i_fK   # in [A/F]
            
        return [d_Xf], i_f, i_fNa, i_fK

    
    
class INaCa():
    '''
    INaCa :: Sodium/Calcium exchange current
    page 12
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    

        #: Maximum conductance
        self.Gncx = 0.0008   # 'Endocardial' : 0.0008  ,   'Epicardial' : 0.0008 * 1.1   'Mid-myocardial' : 0.0008 * 1.4
        self.G_adj = 1.0
                     
    def calculate(self, V, Nai, Cai):
        
        self.kna1   = 15.0        
        self.kna2   = 5.0        
        self.kna3   = 88.12        
        self.kasymm = 12.5        
        self.wna    = 6.0e4
        self.wca    = 6.0e4
        self.wnaca  = 5.0e3 
        self.kcaon  = 1.5e6        
        self.kcaoff = 5.0e3        
        qna = 0.5224
        qca = 0.1670
        self.hca    = exp(qca * V * self.phys.FRT)
        self.hna    = exp(qna * V * self.phys.FRT)
        
        # Parameters h
        h1  = 1.0 + Nai / self.kna3 * (1 + self.hna)
        h2  = (Nai * self.hna) / (self.kna3 * h1)
        h3  = 1.0 / h1
        h4  = 1.0 + Nai / self.kna1 * (1 + Nai / self.kna2)
        h5  = Nai * Nai / (h4 * self.kna1 * self.kna2)
        h6  = 1.0 / h4
        h7  = 1.0 + self.extra.Nao / self.kna3 * (1 + 1 / self.hna)
        h8  = self.extra.Nao / (self.kna3 * self.hna * h7)
        h9  = 1.0 / h7
        h10 = self.kasymm + 1 + self.extra.Nao / self.kna1 * (1 + self.extra.Nao / self.kna2)
        h11 = self.extra.Nao * self.extra.Nao / (h10 * self.kna1 * self.kna2)
        h12 = 1.0 / h10
        
        # Parameters k
        k1   = h12 * self.extra.Cao * self.kcaon
        k2   = self.kcaoff
        k3p  = h9 * self.wca
        k3pp = h8 * self.wnaca
        k3   = k3p + k3pp
        k4p  = h3 * self.wca / self.hca
        k4pp = h2 * self.wnaca
        k4   = k4p + k4pp
        k5   = self.kcaoff
        k6   = h6 * Cai * self.kcaon
        k7   = h5 * h2 * self.wna
        k8   = h8 * h11 * self.wna
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        KmCaAct = 150.0e-6
        allo    = 1 / (1 + (KmCaAct / Cai)**2.0)
        JncxNa  = 3 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
        JncxCa  = E2 * k2 - E1 * k1  

        return 0.8 * self.Gncx*self.G_adj * allo * (self.phys.zna * JncxNa + self.phys.zca * JncxCa)  # Sodium/Calcium exchange current  in [uA/uF]
    
class INaCa_ss():
    '''
    INaCa_ss :: Sodium/Calcium exchanger current into the L-type subspace
    Page 12
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    
            
    def calculate(self, Na_ss, cass, inaca):
        
        h1  = 1.0 + Na_ss / inaca.kna3 * (1 + inaca.hna)
        h2  = (Na_ss * inaca.hna)/(inaca.kna3 * h1)
        h3  = 1 / h1
        h4  = 1 + Na_ss / inaca.kna1 * (1 + Na_ss / inaca.kna2)
        h5  = Na_ss * Na_ss /(h4 * inaca.kna1 * inaca.kna2)
        h6  = 1 / h4
        h7  = 1 + self.extra.Nao / inaca.kna3 * (1 + 1 / inaca.hna)
        h8  = self.extra.Nao / (inaca.kna3 * inaca.hna * h7)
        h9  = 1/h7
        h10 = inaca.kasymm + 1 + self.extra.Nao / inaca.kna1 * (1 + self.extra.Nao / inaca.kna2)
        h11 = self.extra.Nao * self.extra.Nao / (h10 * inaca.kna1 * inaca.kna2)
        h12 = 1/h10
        k1   = h12 * self.extra.Cao * inaca.kcaon
        k2   = inaca.kcaoff
        k3p  = h9 * inaca.wca
        k3pp = h8 * inaca.wnaca
        k3   = k3p + k3pp
        k4p  = h3 * inaca.wca / inaca.hca
        k4pp = h2 * inaca.wnaca
        k4   = k4p + k4pp
        k5   = inaca.kcaoff
        k6   = h6 * cass * inaca.kcaon
        k7   = h5 * h2 * inaca.wna
        k8   = h8 * h11 * inaca.wna
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        KmCaAct = 150.0e-6
        allo    = 1 / (1 + (KmCaAct / cass)**2)
        JncxNa  = 3 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
        JncxCa  = E2 * k2 - E1 * k1
        
        return 0.2 * inaca.Gncx * allo * (self.phys.zna * JncxNa + self.phys.zca * JncxCa)  # Sodium/Calcium exchange current into the T-Tubule subspace  in [uA/uF]
    
class INaK():
    '''
    INaK :: Sodium/Potassium ATPase current
    Page 14
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    

        self.Pnak = 30   # 'Endocardial' : Pnak_b = 30   ,   'Epicardial' : Pnak_b*0.9  'Mid-myocardial' : Pnak_b*0.7
        self.G_adj = 1.0
            
    def calculate(self, V, Nai, Na_ss, Ki, K_ss):
        
        k1p = 949.5
        k1m = 182.4
        k2p = 687.2
        k2m = 39.4
        k3p = 1899.0
        k3m = 79300.0
        k4p = 639.0
        k4m = 40.0
        Knai0 = 9.073
        Knao0 = 27.78
        delta = -0.1550
        Knai = Knai0 * exp(delta * V * self.phys.FRT / 3.0)
        Knao = Knao0 * exp((1.0-delta) * V * self.phys.FRT / 3.0)
        Kki    = 0.5
        Kko    = 0.3582
        MgADP  = 0.05
        MgATP  = 9.8
        Kmgatp = 1.698e-7
        H      = 1.0e-7
        eP     = 4.2
        Khp    = 1.698e-7
        Knap   = 224.0
        Kxkur  = 292.0
        P = eP / (1 + H / Khp + Nai / Knap + Ki / Kxkur)
        a1 = (k1p * (Nai / Knai)**3) / ((1 + Nai / Knai)**3 + (1 + Ki / Kki)**2 - 1)
        b1 = k1m * MgADP
        a2 = k2p
        b2 = (k2m * (self.extra.Nao / Knao)**3) / ((1 + self.extra.Nao / Knao)**3 + (1 + self.extra.Ko / Kko)**2 - 1)
        a3 = (k3p * (self.extra.Ko / Kko)**2  ) / ((1 + self.extra.Nao / Knao)**3 + (1 + self.extra.Ko / Kko)**2 - 1)
        b3 = (k3m * P * H)/(1 + MgATP / Kmgatp)
        a4 = (k4p * MgATP / Kmgatp) / (1 + MgATP / Kmgatp)
        b4 = (k4m * (Ki / Kki)**2) / ((1 + Nai / Knai)**3 + (1 + Ki / Kki)**2 - 1)
        x1 = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2
        x2 = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4
        x3 = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1
        x4 = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        JnakNa = 3 * (E1 * a3 - E2 * b3)
        JnakK  = 2 * (E4 * b1 - E3 * a1)
                            
        return self.Pnak*self.G_adj * (self.phys.zna * JnakNa + self.phys.zk * JnakK)  # Sodium/Potassium ATPase current    in [uA/uF]
   
class IKb():
    '''
    IKb :: Background Potassium current
    Page 15
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    

        self.GKb = 0.003   # 'Endocardial' : GKb_b = 0.003   ,   'Epicardial' : GKb_b*0.6     'Mid-myocardial' : GKb_b
        self.G_adj = 1.0        
            
    def calculate(self, V, nernst):
        
        xkb = 1.0 / (1.0 + exp(-(V - 14.48) / 18.34))
                            
        return self.GKb*self.G_adj * xkb * (V - nernst.EK)  # Background Potassium current   in [uA/uF]
    
class INab():
    '''
    INab :: Background Sodium current
    Page 15
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    

        self.PNab = 3.75e-10   # 'Endocardial' :   ,   'Epicardial' :    'Mid-myocardial' : 
        self.G_adj = 1.0
            
    def calculate(self, V, Nai):                
        v0 = 0        
        A = self.G_adj*self.PNab * self.phys.FFRT * (Nai * exp(V * self.phys.FRT) - self.extra.Nao) / self.phys.FRT            
        U = self.phys.FRT * (V - v0)
        
        if -1e-7<=U and U<=1e-7: 
            return A*(1.0-0.5*U)   # Background Sodium current   in [uA/uF] <- 2017 version
        else:
            return (A*U)/(exp(U)-1)

        
class ICab():
    '''
    ICab :: Background Calcium current
    Page 15
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    

        self.PCab = 2.5e-8
        self.G_adj = 1.0
            
    def calculate(self, V, Cai):        
        B = 2 * self.phys.FRT
        v0 = 0        
        A = self.G_adj*self.PCab * 4.0 * self.phys.FFRT * (Cai * exp( 2.0 * V * self.phys.FRT) - 0.341 * self.extra.Cao) / B
        U = B * (V - v0)

        if -1e-7<=U and U<=1e-7:  
            return A*(1.0-0.5*U) # Background Calcium current  in [uA/uF] <- 2017 version
        else:
            return (A*U)/(exp(U)-1)
            
    
class IpCa():
    '''
    IpCa :: Sarcolemmal Calcium pump current
    Page 15
    '''
    def __init__(self, phys, cell, extra):      
        self.phys = phys
        self.cell = cell
        self.extra = extra    

        self.GpCa = 0.0005
        self.G_adj = 1.0
            
    def calculate(self, Cai):        
        
        KmCap = 0.0005
        IpCa = self.GpCa*self.G_adj * Cai / (KmCap + Cai)  # Sarcolemmal Calcium pump current   in [uA/uF]
                    
        return IpCa
    
class Ryr():
    '''
    '''
    def __init__(self, phys, cell, extra):
        self.phys = phys
        self.cell = cell
        self.extra = extra  

        # initial values        
        self.Jrelnp      = 2.5e-7           # Jrelnp=0
        self.Jrelp       = 3.12e-7          # Jrelp=0                    

        self.y0 = [self.Jrelnp, self.Jrelp]
        
    def diff_eq(self, V, Jrelnp, Jrelp, Ca_jsr, ICaL, camk):

        bt=4.75
        a_rel=0.5*bt        
        Jrel_inf_temp = -a_rel * ICaL / (1 + (1.5 / Ca_jsr)**8)
        self.Jrel_inf = Jrel_inf_temp
        if (self.cell.mode==2):   
            self.Jrel_inf = Jrel_inf_temp * 1.7      
        tau_rel_temp = bt / (1.0 + 0.0123 / Ca_jsr)
        tau_rel = tau_rel_temp
        if (tau_rel_temp < 0.001):   tau_rel = 0.001                        
        d_Jrelnp = (self.Jrel_inf - Jrelnp) / tau_rel   

        btp = 1.25*bt
        a_relp = 0.5*btp
        Jrel_temp = -a_relp * ICaL / (1 + (1.5 / Ca_jsr)**8)
        self.Jrel_infp = Jrel_temp
        if self.cell.mode==2:   
            self.Jrel_infp = Jrel_temp * 1.7                
        tau_relp_temp = btp / (1 + 0.0123 / Ca_jsr)
        tau_relp = tau_relp_temp
        if tau_relp_temp < 0.001:   tau_relp = 0.001        
        d_Jrelp = (self.Jrel_infp - Jrelp) / tau_relp     

        Jrel_scaling_factor = 1.0
        self.Jrel = Jrel_scaling_factor * (1.0 - camk.f) * Jrelnp + camk.f * Jrelp # desc: SR Calcium release flux via Ryanodine receptor  in [mmol/L/ms]

        return [d_Jrelnp, d_Jrelp]

class Diff():
    '''
    Diffusion fluxes
    Page 16
    '''
    def __init__(self):
        self.JdiffNa = None
        self.JdiffK  = None
        self.Jdiff   = None

    def calculate(self, Nai, Na_ss, Ki, K_ss, Cai, cass):
        self.JdiffNa = (Na_ss - Nai) / 2.0   # (sodium.Na_ss - sodium.Nai) / 2
        self.JdiffK  = (K_ss  - Ki)  / 2.0  # (potassium.K_ss  - potassium.Ki)  / 2
        self.Jdiff   = (cass - Cai) / 0.2     # (calcium.cass - calcium.Cai) / 0.2

class Serca():
    '''
    Jup :: Calcium uptake via SERCA pump
    Page 17
    '''
    def __init__(self, phys, cell, extra):
        self.phys = phys
        self.cell = cell
        self.extra = extra  
        
    def calculate(self, Cai, Ca_nsr, Ca_jsr, camk):
        self.upScale = 1.0
        if (self.cell.mode == 1):   
            self.upScale = 1.3
        Jupnp = self.upScale * (0.004375 * Cai / (Cai + 0.00092))
        Jupp  = self.upScale * (2.75 * 0.004375 * Cai / (Cai + 0.00092 - 0.00017))        
        Jleak = 0.0039375 * Ca_nsr / 15.0 # in [mmol/L/ms]
        Jup_b = 1.0
        self.Jup = Jup_b * ((1.0 - camk.f) * Jupnp + camk.f * Jupp - Jleak) # desc: Total Ca2+ uptake, via SERCA pump, from myoplasm to nsr in [mmol/L/ms]        
        self.Jtr = (Ca_nsr - Ca_jsr) / 100.0   # desc: Ca2+ translocation from nsr to jsr    in [mmol/L/ms]
        
        
class Sodium():
    '''
    Intracellular Sodium concentrations
    Page 18
    '''
    def __init__(self, phys, cell, extra): 
        self.phys = phys
        self.cell = cell
        self.extra = extra       
        
        # initial values
        self.Nai      = 7.26800449799999981
        self.Na_ss    = 7.26808997699999981
        
        self.y0 = [self.Nai, self.Na_ss]

    def diff_eq(self, Nai : float, Na_ss : float, 
                INa, INaL, INaCa, INaK, INab, ICaNa, INaCa_ss, IFNa,
                diff): 
        cm = 1.0
        INa_tot    = INa + INaL + INab + 3*INaCa + 3*INaK + IFNa
        d_Nai   = -INa_tot * self.cell.AF * cm / self.cell.vmyo + diff.JdiffNa * self.cell.vss / self.cell.vmyo # Intracellular Potassium concentration
       
        INa_ss_tot = ICaNa + 3*INaCa_ss
        d_Na_ss = -INa_ss_tot * self.cell.AF * cm / self.cell.vss - diff.JdiffNa

        return [d_Nai, d_Na_ss]
        
class Potassium():
    '''
    Intracellular Potassium concentrations
    Page 18
    '''
    def __init__(self, phys, cell, extra):        
        self.phys = phys
        self.cell = cell
        self.extra = extra  
                             
        # initial values
        self.Ki    = 1.44655591799999996e2
        self.K_ss  = 1.44655565099999990e2
        
        self.y0 = [self.Ki, self.K_ss]

    def diff_eq(self, Ki, K_ss, 
                Ito, IKr, IKs, IK1, IKb, INaK, i_stim, ICaK, IFK,
                diff):
        cm = 1.0

        IK_tot = Ito + IKr + IKs + IK1 + IKb - 2 * INaK + IFK       
        d_Ki  = -(IK_tot + i_stim) * cm * self.cell.AF / self.cell.vmyo + diff.JdiffK * self.cell.vss / self.cell.vmyo # Intracellular Potassium concentration
        
        IK_ss_tot = ICaK
        d_K_ss = -IK_ss_tot * cm * self.cell.AF / self.cell.vss - diff.JdiffK  # Potassium concentration in the T-Tubule subspace

        return [d_Ki, d_K_ss]
    
class Calcium():
    '''
    Intracellular Calcium concentrations and buffers
    Page 18
    '''
    def __init__(self, phys, cell, extra):        
        self.phys = phys
        self.cell = cell
        self.extra = extra  
        
        # initial values
        self.Cai = 8.6e-5
        self.cass = 8.49e-5  
        self.Ca_nsr = 1.61957453799999995
        self.Ca_jsr = 1.57123401400000007

        self.y0 = [self.Cai, self.cass, self.Ca_nsr, self.Ca_jsr]

    def diff_eq(self, Cai, cass, Ca_nsr, Ca_jsr, 
                IpCa, ICab, INaCa, ICaL, INaCa_ss, 
                ryr, serca, diff):
        cm = 1.0
        cmdnmax_b = 0.05
        cmdnmax = cmdnmax_b  
        if self.cell.mode == 1: 
            cmdnmax = 1.3*cmdnmax_b        
        kmcmdn  = 0.00238
        trpnmax = 0.07
        kmtrpn  = 0.0005
        BSRmax  = 0.047
        KmBSR   = 0.00087
        BSLmax  = 1.124
        KmBSL   = 0.0087
        csqnmax = 10.0
        kmcsqn  = 0.8
        '''
        desc: Intracellular Calcium concentratium
        in [mmol/L]
        '''
        ICa_tot = IpCa + ICab - 2*INaCa 
        a = kmcmdn + Cai
        b = kmtrpn + Cai
        Bcai = 1.0 / (1.0 + cmdnmax * kmcmdn / (a*a) + trpnmax * kmtrpn / (b*b))        
        d_Cai = Bcai * (-ICa_tot * cm * self.cell.Acap / (2*self.phys.F*self.cell.vmyo) - serca.Jup  * self.cell.vnsr / self.cell.vmyo + diff.Jdiff * self.cell.vss / self.cell.vmyo )
                
        '''
        desc: Calcium concentratium in the T-Tubule subspace
        in [mmol/L]
        ''' 
        ICa_ss_tot = ICaL - 2 * INaCa_ss
        a = KmBSR + cass
        b = KmBSL + cass
        Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (a*a) + BSLmax * KmBSL / (b*b))
        d_cass = Bcass * (-ICa_ss_tot * cm * self.cell.Acap / (2*self.phys.F*self.cell.vss) + ryr.Jrel * self.cell.vjsr / self.cell.vss - diff.Jdiff )
            
        '''
        desc: Calcium concentration in the NSR subspace
        in [mmol/L]
        '''         
        d_Ca_nsr = serca.Jup - serca.Jtr * self.cell.vjsr / self.cell.vnsr
            
        '''
        desc: Calcium concentration in the JSR subspace
        in [mmol/L]
        '''        
        a = kmcsqn + Ca_jsr    
        Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (a*a) )                
        d_Ca_jsr = Bcajsr * (serca.Jtr - ryr.Jrel)

        return [d_Cai, d_cass, d_Ca_nsr, d_Ca_jsr]
        
class ORD2017():
    """    
    O'Hara-Rudy CiPA v1.0 (2017)
    """
    def __init__(self, protocol=None, is_exp_artefact=False):
        
        self.name = "ORD2017"
        
        self.is_exp_artefact = is_exp_artefact
        if self.is_exp_artefact:
            self.exp_artefacts = ExperimentalArtefactsThesis()
        
        self.phys = Phys()
        self.cell = Cell(self.phys)        
        self.extra = Extra()
        
        self.current_response_info = mod_trace.CurrentResponseInfo(protocol)
        
        self.protocol = protocol
        
        self.membrane = Membrane()        
        self.stimulus = Stimulus()
        self.nernst = Nernst(self.phys, self.cell, self.extra)
        self.camk = CaMK()  
        
        self.ina = INa(self.phys, self.cell, self.extra)
        self.inal = INaL(self.phys, self.cell, self.extra)
        self.ito = Ito(self.phys, self.cell, self.extra)        
        self.ical = ICaL(self.phys, self.cell, self.extra)                
        self.ikr = IKr(self.phys, self.cell, self.extra)
        self.iks = IKs(self.phys, self.cell, self.extra)
        self.ik1 = IK1(self.phys, self.cell, self.extra)
        self.ifunny = IFunny(self.phys, self.cell, self.extra)
        
        self.inaca = INaCa(self.phys, self.cell, self.extra)
        self.inacass = INaCa_ss(self.phys, self.cell, self.extra)
        self.inak = INaK(self.phys, self.cell, self.extra)
        self.ikb = IKb(self.phys, self.cell, self.extra)
        self.inab = INab(self.phys, self.cell, self.extra)
        self.icab = ICab(self.phys, self.cell, self.extra)
        self.ipca = IpCa(self.phys, self.cell, self.extra)
                
        self.ryr = Ryr(self.phys, self.cell, self.extra)
        self.serca = Serca(self.phys, self.cell, self.extra)
        self.diff = Diff()
                
        self.sodium = Sodium(self.phys, self.cell, self.extra)
        self.potassium = Potassium(self.phys, self.cell, self.extra)      
        self.calcium = Calcium(self.phys, self.cell, self.extra)

        self.y0 = self.membrane.y0 + self.sodium.y0 + self.potassium.y0 + self.calcium.y0 +\
                    self.ina.y0 + self.inal.y0 + self.ito.y0 + self.ical.y0 + self.ikr.y0 + self.iks.y0 + self.ik1.y0 +\
                        self.ryr.y0 + self.camk.y0 +\
                            self.ifunny.y0 
        if self.is_exp_artefact:
            self.y0 += [0, 0, 0 ,0, 0]
                            
        self.params = []

    def set_result(self, t, y, log=None):
        self.times =  t
        self.V = y[0]         

    def differential_eq(self, t, y):   # len(y) = 41
  
        V, Nai, Na_ss, Ki, K_ss, Cai, cass, Ca_nsr, Ca_jsr,\
            m, hf, hs, j, hsp, jp, \
                mL, hL, hLp, \
                    a, iF, iS, ap, iFp, iSp, \
                        d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp,\
                            IC1, IC2, C1, C2, O, IO, IObound, Obound, Cbound, D,\
                                xs1, xs2, \
                                    xk1, \
                                        Jrelnp, Jrelp, CaMKt, \
                                            Xf = y[:50]
        if self.is_exp_artefact:
            v_p, v_clamp, i_out, v_cmd, v_est = y[50:]

        # Calculate Nernst  
        self.nernst.calculate(Nai, Ki)        
        
        # CaMKt
        d_CaMKt_li = self.camk.d_CaMKt(CaMKt, cass)
      
        # currents  
        d_INa_li, INa = self.ina.diff_eq(V, m, hf, hs, j, hsp, jp, self.camk, self.nernst)
        d_INaL_li, INaL = self.inal.diff_eq(V, mL, hL, hLp, self.camk, self.nernst, self.ina)
        d_Ito_li, Ito = self.ito.diff_eq(V, a, iF, iS, ap, iFp, iSp, self.camk, self.nernst)
        d_ICaL_li, ICaL, ICaNa, ICaK = self.ical.diff_eq(V, d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp, cass, Na_ss, K_ss, self.camk, self.nernst)
        d_IKr_li, IKr = self.ikr.diff_eq(V, IC1, IC2, C1, C2, O, IO, IObound, Obound, Cbound, D, self.camk, self.nernst)
        d_IKs_li, IKs = self.iks.diff_eq(V, xs1, xs2, Cai, self.camk, self.nernst) 
        d_IK1_li, IK1 = self.ik1.diff_eq(V, xk1, self.camk, self.nernst) 
        d_IF_li, i_f, i_fNa, i_fK = self.ifunny.diff_eq(V, Xf, self.camk, self.nernst) 

        i_f = 0             #################################3
        i_fNa = 0           #################################3
        i_fK = 0            #################################3

        # print(t, IK1)
        INaCa = self.inaca.calculate(V, Nai, Cai)
        INaCa_ss = self.inacass.calculate(Na_ss, cass, self.inaca)
        INaK = self.inak.calculate(V, Nai, Na_ss, Ki, K_ss)
        IKb = self.ikb.calculate(V, self.nernst)
        INab = self.inab.calculate(V, Nai)
        ICab = self.icab.calculate(V, Cai)
        IpCa = self.ipca.calculate(Cai)


        d_ryr_li = self.ryr.diff_eq(V, Jrelnp, Jrelp, Ca_jsr, ICaL, self.camk)
        self.diff.calculate(Nai, Na_ss, Ki, K_ss, Cai, cass)        
        self.serca.calculate(Cai, Ca_nsr, Ca_jsr, self.camk)        
        
        d_sodium_li = self.sodium.diff_eq( Nai, Na_ss, 
                                           INa, INaL, INaCa, INaK, INab, ICaNa, INaCa_ss, i_fNa,
                                           self.diff)

        d_potassium_li = self.potassium.diff_eq(Ki, K_ss, 
                                                Ito, IKr, IKs, IK1, IKb, INaK, self.stimulus.I, ICaK, i_fK,
                                                self.diff)

        d_calcium_li = self.calcium.diff_eq(Cai, cass, Ca_nsr, Ca_jsr, 
                                            IpCa, ICab, INaCa, ICaL, INaCa_ss, 
                                            self.ryr, self.serca, self.diff)   
       
        # Membrane potential       
        
        i_K1_ishi = 0
        i_up = 0
        i_leak = 0
        i_no_ion = 0
        
        # -------------------------------------------------------------------
        # Experimental Artefact
        if self.is_exp_artefact:
            ##############Involved##########################            
            i_ion = self.exp_artefacts.c_m*(INa + INaL + INaCa + INaK + INab + ICaNa + INaCa_ss + IpCa + ICab + ICaL + Ito + IKr + IKs + IK1 + IKb + ICaK + i_f + i_K1_ishi + i_no_ion + self.stimulus.I)
            g_leak = self.exp_artefacts.g_leak
            e_leak = self.exp_artefacts.e_leak
            c_m = self.exp_artefacts.c_m
            c_m_star = self.exp_artefacts.c_m_star
            r_access = self.exp_artefacts.r_access
            v_off = self.exp_artefacts.v_off
            tau_clamp = self.exp_artefacts.tau_clamp
            comp_rs = self.exp_artefacts.comp_rs
            comp_predrs = self.exp_artefacts.comp_predrs
            r_access_star = self.exp_artefacts.r_access_star
            tau_sum = self.exp_artefacts.tau_sum
            c_p = self.exp_artefacts.c_p
            c_p_star = self.exp_artefacts.c_p_star
            tau_z = self.exp_artefacts.tau_z

            # y[23] : v_p
            # y[24] : v_clamp
            # y[25] : I_out 
            # y[26] : v_cmd
            # y[27] : v_est

            v_m = V
            # v_p = y[23]
            # v_clamp = y[24]
            # i_out = y[25]
            # v_cmd = y[26]
            # v_est = y[27]

            i_seal_leak = g_leak * (v_m - e_leak)

            #REMOVE to get thesis version
            v_p = v_cmd + r_access_star * comp_rs * (i_ion + i_seal_leak)

            dvm_dt = (1/r_access/c_m) * (v_p + v_off - V) - (i_ion + i_seal_leak) / c_m 

            #dvp_dt = (v_clamp - v_p) / tau_clamp

            #if comp_predrs < .05:
            #    dvest_dt = 0
            #else:
            #    dvest_dt = (v_cmd - v_est) / ((1 - comp_predrs) *
            #            r_access_star * c_m_star / comp_predrs)

            #vcmd_prime = v_cmd + ((comp_rs * r_access_star * i_out) +
            #        (comp_predrs * r_access_star * c_m_star * dvest_dt))

            #dvclamp_dt = (vcmd_prime - v_clamp) / tau_sum

            #i_cp = c_p * dvp_dt - c_p_star * dvclamp_dt
            #
            #if r_access_star < 1E-6:
            #    i_cm = c_m_star * dvclamp_dt
            #else:
            #    i_cm = c_m_star * dvest_dt

            #i_in = (v_p - v_m + v_off) / r_access #+ i_cp - i_cm

            #di_out_dt = (i_in - i_out) / tau_z

            d_V = dvm_dt
            dvp_dt = 0.0
            dvclamp_dt = 0.0
            di_out_dt = 0.0
            dvcmd_dt = 0.0
            dvest_dt = 0.0           

            #d_y[23] = dvp_dt
            #d_y[24] = dvclamp_dt
            #d_y[25] = di_out_dt
            #d_y[27] = dvest_dt

            i_ion = i_ion / self.exp_artefacts.c_m
            i_seal_leak = i_seal_leak / self.exp_artefacts.c_m
            #i_out = i_out / self.exp_artefacts.c_m
            #REMOVE TO GET THESIS VERSION
            i_out = i_ion + i_seal_leak
            #i_cm = i_cm / self.exp_artefacts.c_m
            #i_cp = i_cp / self.exp_artefacts.c_m

            ################################################
            ################################################
  
            if self.current_response_info:
                current_timestep = [
                    mod_trace.Current(name='I_Na', value=INa),
                    mod_trace.Current(name='I_NaL', value=INaL),                
                    mod_trace.Current(name='I_To', value=Ito),
                    mod_trace.Current(name='I_CaL', value=ICaL),
                    mod_trace.Current(name='I_CaNa', value=ICaNa),
                    mod_trace.Current(name='I_CaK', value=ICaK),
                    mod_trace.Current(name='I_Kr', value=IKr),
                    mod_trace.Current(name='I_Ks', value=IKs),
                    mod_trace.Current(name='I_K1', value=IK1),
                    mod_trace.Current(name='I_K1_Ishi', value=i_K1_ishi),                    
                    mod_trace.Current(name='I_NaCa', value=INaCa),
                    mod_trace.Current(name='I_NaCa_ss', value=INaCa_ss),
                    mod_trace.Current(name='I_NaK', value=INaK),
                    mod_trace.Current(name='I_Kb', value=IKb),
                    mod_trace.Current(name='I_Nab', value=INab),
                    mod_trace.Current(name='I_Cab', value=ICab),
                    mod_trace.Current(name='I_pCa', value=IpCa),             
                    mod_trace.Current(name='I_F', value=i_f),        
                    mod_trace.Current(name='I_up', value=i_up),
                    mod_trace.Current(name='I_leak', value=i_leak),       
                    mod_trace.Current(name='I_ion', value=i_ion),
                    mod_trace.Current(name='I_seal_leak', value=i_seal_leak),
                    #trace.Current(name='I_Cm', value=i_cm),
                    #trace.Current(name='I_Cp', value=i_cp),
                    #trace.Current(name='I_in', value=i_in),
                    mod_trace.Current(name='I_out', value=i_out),
                    mod_trace.Current(name='I_no_ion', value=i_no_ion),
                    
                    # trace.Current(name='I_bNa', value=i_b_Na),
                    # trace.Current(name='I_bCa', value=i_b_Ca),
                    # trace.Current(name='I_CaT', value=i_CaT),
                    
                       
                    
                ]
                self.current_response_info.currents.append(current_timestep)

            return [d_V] + d_sodium_li + d_potassium_li + d_calcium_li + \
                        d_INa_li + d_INaL_li + d_Ito_li + d_ICaL_li + d_IKr_li + d_IKs_li + d_IK1_li + \
                            d_ryr_li + d_CaMKt_li + \
                                d_IF_li + [dvp_dt, dvclamp_dt, di_out_dt, dvcmd_dt, dvest_dt]

        # --------------------------------------------------------------------
        # Calculate change in Voltage and Save currents
        else:
            # Membrane potential                 
            I_ion = INa + INaL + INaCa + INaK + INab + ICaNa + INaCa_ss + IpCa + ICab + ICaL + Ito + IKr + IKs + IK1 + IKb + ICaK + i_f + i_K1_ishi + i_no_ion           
            d_V_li = self.membrane.d_V( I_ion + self.stimulus.I )
            
            if self.current_response_info:  # 'INa', 'INaL', 'Ito', 'ICaL', 'IKr', 'IKs', 'IK1'
                current_timestep = [                
                    mod_trace.Current(name='I_Na', value=INa),
                    mod_trace.Current(name='I_NaL', value=INaL),                
                    mod_trace.Current(name='I_To', value=Ito),
                    mod_trace.Current(name='I_CaL', value=ICaL),
                    mod_trace.Current(name='I_CaNa', value=ICaNa),
                    mod_trace.Current(name='I_CaK', value=ICaK),
                    mod_trace.Current(name='I_Kr', value=IKr),
                    mod_trace.Current(name='I_Ks', value=IKs),
                    mod_trace.Current(name='I_K1', value=IK1),
                    mod_trace.Current(name='I_NaCa', value=INaCa),
                    mod_trace.Current(name='I_NaCa_ss', value=INaCa_ss),
                    mod_trace.Current(name='I_NaK', value=INaK),
                    mod_trace.Current(name='I_Kb', value=IKb),
                    mod_trace.Current(name='I_Nab', value=INab),
                    mod_trace.Current(name='I_Cab', value=ICab),
                    mod_trace.Current(name='I_pCa', value=IpCa),    
                    mod_trace.Current(name='I_F', value=i_f),
                    mod_trace.Current(name='I_up', value=i_up),     
                    mod_trace.Current(name='I_leak', value=i_leak),
                    mod_trace.Current(name='I_no_ion', value=i_no_ion),
                    mod_trace.Current(name='I_stim', value=-self.stimulus.I)   

                    # trace.Current(name='I_bNa', value=i_b_Na),
                    # trace.Current(name='I_bCa', value=i_b_Ca),
                    # trace.Current(name='I_CaT', value=i_CaT),
                                                            
                ]
                self.current_response_info.currents.append(current_timestep)
                            
            return d_V_li + d_sodium_li + d_potassium_li + d_calcium_li + \
                        d_INa_li + d_INaL_li + d_Ito_li + d_ICaL_li + d_IKr_li + d_IKs_li + d_IK1_li + \
                            d_ryr_li + d_CaMKt_li + \
                                d_IF_li


    
    def response_diff_eq(self, t, y):
        
        if isinstance(self.protocol, protocol_lib.PacingProtocol)  :                      
            face = self.protocol.pacing(t)
            self.stimulus.cal_stimulation(face) # Stimulus    
        else:              
            if self.is_exp_artefact:
                y[-2] = self.protocol.get_voltage_at_time(t)
            else:
                y[0] = self.protocol.get_voltage_at_time(t)
                    
        return self.differential_eq(t, y)


    def diff_eq_solve_ivp(self, t, y):
        return self.response_diff_eq(t, y)
        
    def diff_eq_odeint(self, y, t, *p):
        return self.response_diff_eq(t, y)



    def change_cell(self, mode):
        self.cell.mode = mode
        
        if self.cell.mode == 0:     # Endocarial
            self.ina.GNa = 75.0   
            self.inal.GNaL = 0.0075 * 2.661  
            self.ito.Gto = 0.02
            self.ical.PCa = 0.0001 * 1.007
            self.ikr.GKr = 4.65854545454545618e-2 # [mS/uF]
            self.iks.GKs = 0.0034 * 1.87
            self.ik1.GK1 = 0.1908* 1.698
            self.inaca.Gncx = 0.0008
            self.inak.Pnak = 30
            self.ikb.GKb = 0.003
            self.inab.PNab = 3.75e-10
            self.icab.PCab = 2.5e-8
            self.ipca.GpCa = 0.0005

            self.ina.G_adj = 1  
            self.inal.G_adj = 1
            self.ito.G_adj = 1
            self.ical.G_adj = 1
            self.ikr.G_adj = 1
            self.iks.G_adj = 1
            self.ik1.G_adj = 1
            self.inaca.G_adj = 1
            self.inak.G_adj = 1
            self.ikb.G_adj = 1
            self.inab.G_adj = 1
            self.icab.G_adj = 1
            self.ipca.G_adj = 1

        elif self.cell.mode == 1:   # Epicardial 
            self.ina.GNa = 75.0   
            self.inal.GNaL = 0.0075 * 2.661  
            self.ito.Gto = 0.02
            self.ical.PCa = 0.0001 * 1.007
            self.ikr.GKr = 4.65854545454545618e-2 # [mS/uF]
            self.iks.GKs = 0.0034 * 1.87
            self.ik1.GK1 = 0.1908* 1.698
            self.inaca.Gncx = 0.0008
            self.inak.Pnak = 30
            self.ikb.GKb = 0.003
            self.inab.PNab = 3.75e-10
            self.icab.PCab = 2.5e-8
            self.ipca.GpCa = 0.0005

            self.ina.G_adj = 1  
            self.inal.G_adj = 0.6
            self.ito.G_adj = 4
            self.ical.G_adj = 1.2
            self.ikr.G_adj = 1.3
            self.iks.G_adj = 1.4
            self.ik1.G_adj = 1.2
            self.inaca.G_adj = 1.1
            self.inak.G_adj = 0.9
            self.ikb.G_adj = 0.6
            self.inab.G_adj = 1
            self.icab.G_adj = 1
            self.ipca.G_adj = 1

        elif self.cell.mode == 2:   # Mid-myocardial
            self.ina.GNa = 75.0   
            self.inal.GNaL = 0.0075 * 2.661  
            self.ito.Gto = 0.02
            self.ical.PCa = 0.0001 * 1.007
            self.ikr.GKr = 4.65854545454545618e-2 # [mS/uF]
            self.iks.GKs = 0.0034 * 1.87
            self.ik1.GK1 = 0.1908* 1.698
            self.inaca.Gncx = 0.0008
            self.inak.Pnak = 30
            self.ikb.GKb = 0.003
            self.inab.PNab = 3.75e-10
            self.icab.PCab = 2.5e-8
            self.ipca.GpCa = 0.0005

            self.ina.G_adj = 1  
            self.inal.G_adj = 1
            self.ito.G_adj = 4
            self.ical.G_adj = 2.5
            self.ikr.G_adj = 0.8
            self.iks.G_adj = 1
            self.ik1.G_adj = 1.3
            self.inaca.G_adj = 1.4
            self.inak.G_adj = 0.7
            self.ikb.G_adj = 1
            self.inab.G_adj = 1
            self.icab.G_adj = 1
            self.ipca.G_adj = 1

        elif self.cell.mode == 3:   # Max
            f = 1.5
            self.ina.GNa = 75.0 * 1 * f  
            self.inal.GNaL = 0.0075 * 2.661 * 1 * f  
            self.ito.Gto = 0.02 * 4 * f
            self.ical.PCa = 0.0001 * 1.007 * 2.5 * f  
            self.ikr.GKr = 4.65854545454545618e-2 * 1.3 * f
            self.iks.GKs = 0.0034 * 1.87 * 1.4 * f
            self.ik1.GK1 = 0.1908 * 1.698 * 1.3 * f
            self.inaca.Gncx = 0.0008 * 1.4
            self.inak.Pnak = 30 * 1
            self.ikb.GKb = 0.003 * 1
            self.inab.PNab = 3.75e-10 * 1
            self.icab.PCab = 2.5e-8 * 1
            self.ipca.GpCa = 0.0005 * 1    

            self.ina.G_adj = 1  
            self.inal.G_adj = 1
            self.ito.G_adj = 1
            self.ical.G_adj = 1
            self.ikr.G_adj = 1
            self.iks.G_adj = 1
            self.ik1.G_adj = 1
            self.inaca.G_adj = 1
            self.inak.G_adj = 1
            self.ikb.G_adj = 1
            self.inab.G_adj = 1
            self.icab.G_adj = 1
            self.ipca.G_adj = 1

        
        
       





def main():
    start_time = time.time()
    ord2017 = ORD2017(None)    
    print("--- %s seconds ---"%(time.time()-start_time))


if __name__ == '__main__':
    main()
