#!/usr/bin/env python3

"""
plumeequations.py
    Calculates the rise height of a negatively buoyant jet in an environment 
    with arbitrary stratification.  

    The ode system to solve is similar to that defined by Morton, Turner & 
    Taylor (1956) for volume flux, q = b**2 * u and momentum flux, 
    m = b**2 * u**2.  However, there is no explicit equation for buoyancy 
    conservation.  Instead, buoyancy is accounted for through the g' term, which    is calculated at each altitude, x, through a separate function.
    flux of buoyancy, f = f0.
      => b = q / np.sqrt(m)
         u = m / q
    dq/dx = 2 * alphae * np.sqrt(m)
    dm/dx = b**2 * g' = q**2 * g' / m = f0 * q / m
"""

from scipy.integrate import solve_ivp
from SW_Properties.SW_Density import (rho_sw, drho_sw_ds,
                                      drho_sw_dT, rho_plain)

import numpy as np



def derivs(x, V, params):
    ''' Order of params: 
    entrainment_coefficient,
    settling_velocity, 
    stratification_type, 
    gradient/step_height, 
    density_base, 
    density_top,
    density_particle,
    initial_conditions,
    '''
    Q, M, P  = V
    alpha = params[0]
    u_s   = params[1]
    gp    = gprimed(x, V, params)
    
    dVdx    = np.zeros_like(V)
    dVdx[0] = 2 * alpha * np.sqrt(M)
    dVdx[1] = gp * Q**2 / M
    dVdx[2] = 2 * Q / np.sqrt(M) * u_s
    
    return dVdx


def gprimed(x, V, params):
    rho_0 = params[4]
    rho_w = params[5]
    rho_p = params[6]
    V0    = params[7]
    g     = params[8]
    
    Q0, M0, P0 = V0
    phi_0 = 0
    if P0 != 0:
        phi_0 = P0 / Q0

    Q, M, P = V
    phi = np.zeros_like(P)   # particle concentration
    if np.any(P != 0):
        phi = P / Q

    rho_a = rho_amb(x, V, params)
    rho_b = rho_bulk(x, V, params)

    return (rho_a - rho_b) / rho_0 * g


def rho_bulk(x, V, params):
    '''
    rho_b = (1 - phi) * rho_w + phi * rho_s
   
    but as rho_w(z) = Q_0/Q(z) * (1 - phi_0) / (1 - phi(z)) * rho_{w,0},
    => rho_b = (1 - phi_0) * Q_0/Q(z) * rho_{w,0} + phi(z) * rho_s
    
    '''
    Q0, M0, P0 = params[7]
    phi_0 = 0
    if P0 != 0:
        phi_0 = P0 / Q0

    Q, M, P = V
    phi = 0
    if np.any(P != 0):
        phi = P / Q

    rho_0 = params[4]
    rho_w = params[5]
    rho_s = params[6]

    return (1 - phi_0) * Q0 / Q * rho_w + phi * rho_s


def rho_amb(x, V, params):
    ''' Call sign for params: 
    (alpha, stratification_type, gradient/step height, density at base, 
    density at top)
    '''
    
    stratification_type = params[2]
    if stratification_type == 'gradient':
        gradient = params[3]
        rho_base = params[4]
        rho_a = rho_base + x * gradient
    else:
        step_height = params[2]
        rho_base    = params[3]
        rho_top     = params[4]
        rho_a = np.where(x < step_height, rho_base, rho_top)
    return rho_a


def woods2010_derivs(x, V, params=(.1, 1)):
    alpha, N2 = params

    Q, M, F = np.array(V)  # structure of soln from solve_ivp is 3xN
    dVdx    = np.zeros_like(V)
    dVdx[0] = 2 * alpha * np.sqrt(M)
    dVdx[1] = F * Q / M
    dVdx[2] = -N2 * Q

    return dVdx


def params_from_dict(expt_conds):

    params = []
    if 'alpha_e' in expt_conds:
        params.append(expt_conds['alpha_e'])
    else:
        params.append(0.1)
    if 'N2' in expt_conds:
        params.append(expt_conds['N2'])
    else:
        params.append(1.0)
    if 'stratification type' in expt_conds:
        params.append(expt_conds['stratification type'])
    else:
        params.append('gradient')
    if 'phi_0' in expt_conds:
        params.append(expt_conds['phi_0'])
    else:
        params.append(0)
    if 'rho_b' in expt_conds:
        params.append(expt_conds['rho_b'])
    else:
        params.append(1.01)
    if 'rho_0' in expt_conds:
        params.append(expt_conds['rho_0'])
    else:
        params.append(1.001)
    if 'gp0' in expt_conds:
        params.append(expt_conds['gp0'])
    else:
        params.append(-1)
    

    return tuple(params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plot = True

    alpha_e = .1      # entrainement coefficient
    N2      = .684    # buoyancy frequency/[Hz]
    p       = (alpha_e, N2)
    g       = 981     # gravitational field strength/[cm/s2]
    t0      =  0.     # vent height/[cm]
    t1      = 50.     # max height of domain/[cm]
    phi_0   = 0.050   # particle volume fraction at source/[-]
    rho_w   =  .9998  # density of water within plume/[g/cm3]
    rho_0   = 1.0010  # ambient density at source/[g/cm3]
    rho_p   = 2.5     # density of particles/[g/cm3]
    r0 =    .4        # source radius/[cm]
    Q0 =  30.         # source volume flux/[cm3/s]
    u0 = Q0 / (np.pi * r0**2) # source velocity/[cm/s]
    us = 0            # settling velocity/[cm/s]
    M0 = Q0 * u0      # source momentum flux/[cm4/s2]
    P0 = phi_0 * Q0   # source particle volume flux/[cm3/s]
    V0 = (Q0, M0, P0)
    #sol = solve_ivp(derivs, [t0, t1], V0)
    stratification_type = 'gradient'  # 'gradient' or step height value
    gradient = -1.7333e-4
    params   = (alpha_e, us, stratification_type, 
                gradient, rho_0, rho_w, rho_p, V0, g)
    rho_b    = rho_bulk(0, V0, params)  # source bulk density/[g/cm3]
    gp0      = gprimed(0, V0, params)   # reduced gravity at source/[cm/s2]

    expt_conds = {'alpha_e': alpha_e,
                  'N2': N2,
                  'stratification type': stratification_type,
                  'gradient': gradient,
                  'phi_0': phi_0,
                  'rho_b': rho_b,
                  'rho_0': rho_0,
                  'gp0': gp0,
                  'r0': r0,
                  'Q0': Q0,
                  'u0': u0,
                  'us': us,
                  'V0': V0}

    print(f'alphae = {expt_conds["alpha_e"]}')
    print(f'N2     = {expt_conds["N2"]}')
    print(f'phi_0  = {expt_conds["phi_0"]}')
    print(f'rho_b  = {expt_conds["rho_b"]:.3f}')
    print(f'gp0    = {expt_conds["gp0"]:.3f}')
    print(f'Q0     = {expt_conds["Q0"]}')
    print(f'u0     = {expt_conds["u0"]:.3f}')
    print(f'us     = {expt_conds["us"]:.3f}')

    n_steps = 501
    t  = np.linspace(t0, t1, n_steps)
    sol = solve_ivp(derivs, [t0, t1], V0, t_eval=t, args=(params,))

    if plot_soln:
        # Plot solution
        plt.close('all')
        plt.plot(sol.y.T, sol.t, '-')
        plt.legend(('Q', 'M', 'P'))
        plt.grid('both')

        Q, M, P  = sol.y
        b, u, phi = Q / np.sqrt(np.pi * M), M / Q, P / Q
        gp    = gprimed(sol.t, sol.y, params)
        rho_a = rho_amb(sol.t, sol.y, params)
        rho_b = rho_bulk(sol.t, sol.y, params)

        W  = np.vstack([b, u/u0, phi,
                        #gp,
                        rho_a, rho_b])
        W0 = W.T[0]
        W0inv = list(map(lambda x: 1/x, W0))
        plt.subplots()
        plt.plot(W.T * W0inv, sol.t, '-')
        plt.xlabel(r'Plume parameter')
        plt.ylabel(r'Altitude, $x$')
        plt.legend((r'$b/b_0$', r'$\bar{u}/\bar{u}_0$', r'$\phi/\phi_0$',
                    r'$\rho_a/\rho_{a,0}$', r'$\rho_b/\rho_{b,0}$'))
        plt.grid('both')

    print(f"Plume predicted to reach {sol.t[-1]:.3f} units)")
