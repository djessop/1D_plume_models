#!/usr/bin/env python3

"""
uniformEnvironmentRiseHeight.py
    Calculates the rise height of a negatively buoyant jet in an uniform (i.e.
    no stratification) environment. 

    The ode system to solve is that defined by Morton, Turner & Taylor (1956) 
    for volume flux, q = b**2 * u, momentum flux, m = b**2 * u**2, and a fixed 
    flux of buoyancy, f = f0.
      => b = q / np.sqrt(m)
         u = m / q
        g' = f0 / q
    dq/dx = 2 * alphae * np.sqrt(m)
    dm/dx = b**2 * g' = q**2 * g' / m = f0 * q / m
    df/dx = 0   => f0 = const. = q * g'
"""

from scipy.integrate import ode
from SW_Properties.SW_Density import (rho_sw,
                                      drho_sw_ds,
                                      drho_sw_dT,
                                      rho_plain)

import numpy as np
import matplotlib.pyplot as plt
import pandas


# Global constants
alphae =   0.1  # Entrainment coefficient
g      = 981    # gravitational acceleration, cm/s**2


def derivs(z, state):
    """
    ODE system as per Morton et al. (1956)
    dq/dx = 2 * alphae * np.sqrt(m)
    dm/dx = b**2 * g' = q**2 * g' / m = f0 * q / m
    """
    q = state[0]
    m = state[1]
    return [2 * alphae * np.sqrt(m),
            f0 * q / m]


def solve_system(q0, m0, f0, z_final=50.):
    # Set up the integrator, using vode and backwards differentiation formula
    r = ode(derivs).set_integrator('vode', method='bdf')
    #set_integrator('lsoda')

    # Array of positions at which the solution is sought
    z_start =  0.
    z_final =  z_final
    delta_z =   .1
    # Number of 'time' steps
    num_steps = np.int64(np.floor((z_final - z_start) / delta_z) + 1)

    # Set initial conditions
    M0 = [q0, m0]
    r.set_initial_value(M0, z_start)

    # Storage for solutions
    z    = np.zeros((num_steps))
    M    = np.zeros((num_steps, 2))
    z[0] = z_start
    M[0] = M0

    # Driver for the integration.  Note that the success of the integration
    # is checked at each requested step.
    step = 1
    while r.successful() and step < num_steps:
        r.integrate(r.t + delta_z)
        
        # Store the current solution
        z[step] = r.t
        M[step] = r.y
        step += 1

    return z, M


if __name__ == "__main__":
    # The following nomenclature is used in the calculations below
    #   S - salinity of the plume
    #   S0 - Salinity of the lower layer of the tank
    #   T - Temp. of the plume water, degC
    #   T0 - Temp. of the tank water, degC
    #   rho_b - bulk density of bucket (water + particles), g/cm3
    #   rho_0 - density of lower portion of tank
    
    # Initial conditions
    S  =   0.0      # Salinity of the plume, g/kg or ppt
    S0 =   0.0      # Salinity of the lower layer of the tank
    T  =  23.5      # Temp. of the plume water, degC
    T0 =  23.5      # Temp. of the tank water, degC
    
    rho_b = 1.0020  # g/cm3 rho_sw(T, S)
    rho_0 = 1.0306  # g/cm3 rho_sw(T0, S0)
    rhow  =  .9991  # g/cm3 rho_plain(T0)
    gprimed = g * (rho_0 - rho_b) / rho_0

    r0 =     .4     # cm
    q0 =   20.      # cm3/s
    u0 = q0 / (np.pi * r0**2)
    m0 = q0 * u0 
    #f0 = -413.09
    f0    = gprimed * q0

    print("ICs: \n  q0 = %15.4f \n  m0 = %15.4f \n  f0 = %15.4f \n"
          % (q0, m0, f0))

    # Dataframe of experimental conditions
    # df = pandas.read_csv('/home/david/Experiments/experiments-log-LMV.csv', 
    #                      sep=';')
    # df = pandas.read_excel('/home/david/Experiements/experiments-log-LMV.xls',
    #                        'allData')

    z, M = solve_system(q0, m0, f0)

    # Integration done!  Set "empty" entries in solution vectors to None
    z[1:] = np.where(z[1:] != 0.0, z[1:], None)
    M     = M[~np.isnan(z), :]  # Trim excess entries from solution vector
    z     = z[~np.isnan(z)]
    plot  = False
    
    if plot:
        q, m = M.T
        b = q / np.sqrt(m)
        u = m / q

        # Plot the solutions
        plt.close('all')
        fig = plt.figure()
        plt.plot(b, z, '-', label='width', lw=2)
        plt.plot(u, z, '-', label='speed', lw=2)
        plt.legend() #('q', 'm'), loc='best')
        #plt.xlabel('plume parameter')
        plt.ylabel('Altitude/[cm]')
        plt.xlim((0, 28))
        plt.axhline(z[-1], 0, 1, c='k', ls='--', lw=2, zorder=0)
        plt.grid()
        fig.tight_layout()
        plt.show()

    print(f"Plume predicted to reach {z[-1]:.3f} units")
