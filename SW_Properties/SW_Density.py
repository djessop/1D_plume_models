#!/usr/bin/python
"""
    SW_Density    Density of seawater
     USAGE:  rho = SW_Density(T,uT,S,uS)
    
     DESCRIPTION:
       Density of seawater at atmospheric pressure (0.1 MPa) using Eq. (8)
       given by [1] which best fit the data of [2] and [3]. The pure water
       density equation is a best fit to the data of [4]. 
       Values at temperature higher than the normal boiling temperature are
       calculated at the saturation pressure.
    
     INPUT:
       T  = temperature 
       uT = temperature unit
            'C'  : [degree Celsius] (ITS-90)
            'K'  : [Kelvin]
            'F'  : [degree Fahrenheit] 
            'R'  : [Rankine]
       S  = salinity
       uS = salinity unit
            'ppt': [g/kg]  (reference-composition salinity)
            'ppm': [mg/kg] (in parts per million)
            'w'  : [kg/kg] (mass fraction)
            '%'  : [kg/kg] (in parts per hundred)
       
       Note: T and S must have the same dimensions
    
     OUTPUT:
       rho = density [kg/m^3]
    
       Note: rho will have the same dimensions as T and S
    
     VALIDITY: 0 < T < 180 C; 0 < S < 160 g/kg
     
     ACCURACY: 0.1%
     
     REVISION HISTORY:
       2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
         - Initial version
       2012-06-06: Karan H. Mistry (mistry@mit.edu), MIT
         - Allow T,S input in various units
         - Allow T,S to be matrices of any size
       2016-09-07: D. E. Jessop
         - Rewritten in python
       2024-01-22: DEJ
         - Modifications to documentation 
    
     DISCLAIMER:
       This software is provided "as is" without warranty of any kind.
       See the file sw_copy.m for conditions of use and license.
     
     REFERENCES:
       [1] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
           and Water Treatment, 16, 354-380, 2010. 
           (http://web.mit.edu/seawater/)
       [2] Isdale, and Morris, Desalination, 10(4), 329, 1972.
       [3] Millero and Poisson, Deep-Sea Research, 28A (6), 625, 1981
       [4] IAPWS release on the Thermodynamic properties of ordinary water 
           substance, 1996. 
"""

# Constants for the subfunctions
a = [9.9992293295E+02,    
     2.0341179217E-02,    
    -6.1624591598E-03,    
     2.2614664708E-05,    
    -4.6570659168E-08]

b = [8.0200240891E+02,    
    -2.0005183488E+00,    
     1.6771024982E-02,    
    -3.0600536746E-05,    
    -1.6132224742E-05]


def parse_units():
    return


# Density of plain water
def rho_plain_water(T):
    return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4


# Density increase due to salinity
def delta_rho(T, s=0.0):
    '''
    Salinity units must be in g / kg
    '''
    return (b[0]*s + b[1]*s*T + b[2]*s*T**2 + b[3]*s*T**3 
            + b[4]*s**2*T**2)


def rho_plain(T, S=None, output_units='cgs', fjac=False):
    """
    rho_plain defines estimates the density of plain water as 
    a function of both temperature.

    Inputs:
    T            Temperature in deg C
    S            Salinity in ppt
    output_units  Units system for the output either 'cgs' or 'mks'
    fjac         Boolean switch to return the analytic jacobian (default False)

    Note that the subfunctions use the percentage salinity, whereas the input 
    salinity is in ppt by default.
    """

    rho_w = rho_plain_water(T)
    if output_units == 'cgs':
        rho_w /= 1000
    return rho_w


def rho_sw(T, S=0.0, output_units='cgs', fjac=False):
    """
    rho_sw defines several functions for the estimation of the density water as 
    a function of both temperature and salinity.

    Inputs:
    T            Temperature in deg C
    S            Salinity in ppt
    output_units  Units system for the output either 'cgs' or 'mks'
    fjac         Boolean switch to return the analytic jacobian (default False)

    Note that the subfunctions use the percentage salinity, whereas the input 
    salinity is in ppt by default.
    """

    s = S/1e3
    rho_w  = rho_plain_water(T)
    d_rho  = delta_rho(T, s)
    rho_sw = rho_w + d_rho
    if output_units == 'cgs':
        rho_sw /= 1e3
        rho_w  /= 1e3
    # elif output_units == 'mks':
    #     return (rho_sw, rho_w)
    # else:
    #     return (rho_sw, rho_w)    

    return rho_sw


def drho_plain_water_dT(T):
    return a[1] + 2*a[2]*T + 3*a[3]*T**2 + 4*a[4]*T**3


def drho_plain_water_ds(T):
    return 0.0


def ddelta_rho_dT(T, s=0.0):
    return (b[1]*s + 2*b[2]*s*T + 3*b[3]*s*T**2
        + 2*b[4]*s**2*T)


def ddelta_rho_ds(T, s=0.0, output_arguments='cgs'):
    return (b[0] + b[1]*T + b[2]*T**2 + b[3]*T**3 
            + 2*b[4]*s*T**2)


def drho_sw_dT(T, s=0.0):
    return drho_plain_water_dT(T) + ddelta_rho_dT(T, s)


def drho_sw_ds(T, s=0.0):  
    return ddelta_rho_ds(T, s)


if __name__ == "__main__":
    import numpy as np

    from scipy.optimize import newton, curve_fit, fsolve
    
    
    # BEGIN
    T, S = 20, 30  # °C and g/kg
    rhow  = rho_plain(T, S)
    rhosw = rho_sw(T, S)
    
    def target(S, x, T=20.):
        return rho_sw(T, S) - x

    
    target_density  = 1.01
    target_salinity = newton(target, x0=S, args=(target_density,T))
    print(f"Salinity for target density {target_density} g/cm3: "
          + f"{target_salinity:.4f}")
