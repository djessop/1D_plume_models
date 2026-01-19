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
from scipy.optimize import newton
from sw_properties import (rho_sw, drho_sw_ds,
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
    rhoa_0 = params[4]
    rhoa_t = params[5]
    rho_p  = params[6]
    V0     = params[7]
    g      = params[8]
    
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

    return (rho_a - rho_b) / rhoa_0 * g


def rho_bulk(x, V, params):
    '''
    rho_b = (1 - phi) * rhoa_t + phi * rho_s
   
    but as rhoa(z) = Q_0/Q(z) * (1 - phi_0) / (1 - phi(z)) * rho_{w,0},
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

    rhoa_0 = params[4]
    rhoa_t = params[5]
    rho_s  = params[6]

    return (1 - phi_0) * Q0 / Q * rhoa_t + phi * rho_s


def rho_amb(x, V, params):
    ''' Call sign for params: 
    (alpha, stratification_type, gradient/step height, density at base, 
    density at top)
    '''
    
    stratification_type = params[2]
    if stratification_type == 'uniform':
        rho_a = params[4]  # "rho_base"
    elif stratification_type == 'gradient':
        gradient = params[3]
        rho_base = params[4]
        rho_a = rho_base + x * gradient
    elif stratification_type == 'step':
        step_height = params[3]
        rho_base    = params[4]
        rho_top     = params[5]
        s0          = params[9]
        supper      = params[10]
        s           = np.where(x < step_height, s0, supper)
        rho_a       = rho_sw(20, s)
    else:
        raise NameError('Unknown stratification type')
    return rho_a


def ambient_salinity(x, V, params):
    rho_a = rho_amb(x, V, params)

    T, s0 = 20., 20.

    def target(s, x, T):
        return rho_sw(T, s) - x

    if np.size(rho_a) > 1:# type(rho_a) == np.ndarray or type(rho_a) == list:
        amb_salinity = []
        for target_density in rho_a:
            amb_salinity.append(newton(target, x0=s0,
                                       args=(target_density, T)))
    else:
        amb_salinity = newton(target, x0=s0, args=(rho_a, T))

    return amb_salinity


def woods2010_derivs(x, V, params=(.1, 1)):
    '''
    
    '''
    alpha, N2 = params

    Q, M, F = np.array(V)  # structure of soln from solve_ivp is 3xN
    dVdx    = np.zeros_like(V)
    dVdx[0] = 2 * alpha * np.sqrt(M)
    dVdx[1] = F * Q / M
    dVdx[2] = -N2 * Q

    return dVdx


def derivs_base_parameters(x, V, params=(.1, 1)):
    '''
    The following equations give exactly the same solution as "woods2010_derivs"
    using the {b, u, gp} set of parameters, rather than the {Q, M, F} set.
    '''
    alphae, N2 = params
    b, u, gp   = np.array(V)

    dVdx = np.array([2*alphae - 1/2 * b * gp / u**2,  # N.B. b * gp / u**2 = Ri
                     gp / u   - 2*alphae * u / b,
                     -N2      - 2*alphae * gp / b])
    return dVdxw


def derivs_buphis(x, V, params=(.1, 0, None)):
    '''
    The following equations use the {b, u, phi, s} set of parameters, where the 
    equations for dphi/dx and ds/dx come from conservation of particle and salt
    mass.
    '''
    alphae, us, *_ = params
    b, u, phi, s   = np.array(V)

    sa = s_profile(x, params) # ambient_salinity(x, V, params)
    gp = gprimed_buphis(x, V, params)

    dVdx = np.array([2*alphae - 1/2 * b * gp / u**2,  # N.B. b * gp / u**2 = Ri
                     gp / u   - 2*alphae * u / b,
                     2 / b * (us / u - alphae * phi),
                     2 / b * alphae * (sa - s)])
                     
    return dVdx


def gprimed_buphis(x, V, params):
    rhoa_0 = params[4]
    rhoa_t = params[5]
    rho_p  = params[6]
    V0     = params[7]
    g      = params[8]
    
    rho_a = rho_amb(x, V, params)
    rho   = rho_bulk_buphis(x, V, params)

    return (rho_a - rho) / rho * g


def rho_bulk_buphis(x, V, params):
    '''
    rho_b = (1 - phi) * rhoa + phi * rho_s
   
    but as rho_w(z) = Q_0/Q(z) * (1 - phi_0) / (1 - phi(z)) * rho_{w,0},
    => rho_b = (1 - phi_0) * Q_0/Q(z) * rho_{w,0} + phi(z) * rho_s
    
    '''
    T = 20.
    b, u, phi, s = V

    rho_s = params[6]

    return (1 - phi) * rho_sw(T, s) + phi * rho_s


def convert_bugp_qmf(x, V, params=(.1, 1, None)):
    b, u, gp = np.array(V)
    q, m, f  = b**2 * u, b**2 * u**2, b**2 * u * gp

    return np.array([q, m, f])


def convert_qmf_bugp(x, V, params=(.1, 1, None), include_pi=False):
    q, m, f  = np.array(V)
    b, u, gp = q / np.sqrt(m), m / q, f / q

    return np.array([b, u, gp])


def params_from_dict(expt_conds):
    # to be rewritten taking new order into account
    params = []
    if 'alpha_e' in expt_conds:  # 0
        params.append(expt_conds['alpha_e'])
    else:
        params.append(0.1)
    if 'N2' in expt_conds:       # 1
        params.append(expt_conds['N2'])
    else:
        params.append(1.0)
    if 'stratification_type' in expt_conds:  # 2
        params.append(expt_conds['stratification_type'])
    else:
        params.append('gradient')
    if 'phi_0' in expt_conds:   # 3
        params.append(expt_conds['phi_0'])
    else:
        params.append(0)
    if 'rho_b' in expt_conds:   # 4
        params.append(expt_conds['rho_b'])
    else:
        params.append(1.01)
    if 'rhoa_0' in expt_conds:
        params.append(expt_conds['rhoa_0'])
    else:
        params.append(1.001)
    if 'gp0' in expt_conds:
        params.append(expt_conds['gp0'])
    else:
        params.append(-1)
    

    return tuple(params)


def s_profile(x, params):
    H  = params[3]  # se_conds['step_height']
    s0 = params[9]  # se_conds['s0']
    su = params[10] # se_conds['supper']
    T0 = params[11] 
    
    stratification_type = params[2]
    if stratification_type == 'uniform':
        return rho * s0
    elif stratification_type == 'step':
        return np.where(x <= H, s0, su)
    elif stratification_type == 'gradient':
        return 



def check_se_conds(se_conds_fname):
    import json
    import pandas as pd

    se_conds_file, ext = se_conds_fname.split('.')
    if ext == 'json':
        with open(se_conds_fname, 'r') as f:
            se_conds = json.load(f)
    elif 'xls' in ext:
        se_conds = pd.read_excel(se_conds_fname, index_col=[0])['value']
        se_conds.to_json(se_conds_file + '.json')
        se_conds = se_conds.to_dict()
    else:
        raise NameError('File extension should be .xls* or .json')

    if se_conds['units'].lower() != 'cgs':
        raise ValueError("Please ensure that units are in CGS")
    
    for par in ['r0', 'u0', 'rho0', 'rhoa0', 's0', 'us', 
                'phi0', 'stratification_type']:
        if par not in se_conds.keys():
            raise KeyError(f"key {par} not in se_conds")
        rhoa_type = se_conds['stratification_type']
        if rhoa_type not in ['uniform', 'step', 'gradient']:
            raise KeyError((f"stratification_type {rhoa_type} must be one of") \
                           + (" 'uniform', 'step' or 'gradient'"))
        if rhoa_type == 'step':
            for par in ['step_height', 'rhoa_upper']:
                if par not in se_conds.keys():
                    raise KeyError(f"key {par} not in se_conds")
        elif rhoa_type == 'gradient':
            for par in ['rhoa_gradient', ]:
                if par not in se_conds.keys():
                    raise KeyError(f"key {par} not in se_conds")

    return se_conds


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    import json
    
    plot_soln = False
    se_conds_fname = 'source_environmental_conditions.xlsx'

    if len(sys.argv) > 2:
        plot_soln = bool(sys.argv[2])
    if len(sys.argv) > 3:
        se_conds_fname = sys.argv[3]

    se_conds = check_se_conds(se_conds_fname)

    g       = 981   		  # gravitational field strength/[cm/s2]
    alpha_e = se_conds['alphae']  # entrainement coefficient
    phi0    = se_conds['phi0']    # particle volume fraction at source/[-]
    rhoa0   = se_conds['rhoa0']   # ambient density at source/[g/cm3]
    rho_p   = se_conds['rhop']    # density of particles/[g/cm3]
    r0      = se_conds['r0']      # source radius/[cm]
    u0      = se_conds['u0']      # source velocity/[cm/s]
    us      = se_conds['us']      # settling velocity/[cm/s]
    s0      = se_conds['s0']      # plume salinity at source
    supper  = se_conds['supper']  # plume salinity in upper portion of tank
    uS      = se_conds['uS']      # units of salinity/[g/kg, ppt, kg/kg]
    T0      = se_conds['T0']      # temperature at source
    uT      = se_conds['uT']      # units of temperature/[°C or K]
    Q0      = np.pi * r0**2 * u0  # source volume flux/[cm3/s]
    M0      = Q0 * u0             # source momentum flux/[cm4/s2]
    P0      = phi0 * Q0           # source particle volume flux/[cm3/s]
    V0      = (r0, u0, phi0, s0)  # (Q0, M0, P0)

    stratification_type = se_conds['stratification_type']
    if stratification_type == 'uniform':
        gradient = 0  
        rhoa_0   = se_conds['rhoa0']
        rhoa_top = rhoa_0
        N2       = 0
        supper   = s0
    elif stratification_type == 'step':
        gradient = se_conds['step_height'] # takes role of step_height
        rhoa_0   = rho_sw(T0, s0)     # se_conds['rhoa0']
        rhoa_top = rho_sw(T0, supper) # se_conds['rhoa_upper']
        N2       = g * (rhoa_0 - rhoa_top) 
    elif stratification_type == 'gradient':
        gradient = se_conds['gradient']
        rhoa_0   = se_conds['rhoa0']
        rhoa_top = 0.998
        N2       = - g * gradient   ## Gradient is normalised by either
                                    ## rhoa_0 or s0

    p       = (alpha_e, N2)
    params   = (alpha_e, us, stratification_type, 
                gradient, rhoa_0, rhoa_top, rho_p, V0, g,
                s0, supper, T0)
    rho_b    = rho_bulk_buphis(0, V0, params)  # source bulk density/[g/cm3]
    gp0      = gprimed_buphis(0, V0, params)   # g' at source/[cm/s2]

    se_conds['gp0']    = gp0
    se_conds['rho_b']  = rho_b
    se_conds['rhoa_0'] = rhoa_0
    se_conds['Q0']     = Q0
    se_conds['V0']     = V0
    se_conds['N2']     = N2

    ### SOLVE SYSTEM AND OUTPUT INFORMATION
    print(f'strat = {se_conds["stratification_type"]:>8s}')
    print(f'units  = {se_conds["units"]:>7s}')
    print(f'alphae = {se_conds["alphae"]:7.3f}')
    print(f'r0     = {se_conds["r0"]:7.3f}')
    print(f'u0     = {se_conds["u0"]:7.3f}')
    print(f'us     = {se_conds["us"]:7.3f}')
    print(f'phi0   = {se_conds["phi0"]:7.3f}')
    print(f's0     = {se_conds["s0"]:7.3f}')
    print(f'rhoa_0 = {se_conds["rhoa_0"]:7.3f}')
    print(f'rho_b  = {se_conds["rho_b"]:7.3f}')
    print(f'gp0    = {se_conds["gp0"]:7.3f}')
    print(f'Q0     = {se_conds["Q0"]:7.3f}')
    
    n_steps = 1001
    t0      =   0.    # vent height/[cm]
    t1      = 100.    # max height of domain/[cm]
    t       = np.linspace(t0, t1, n_steps)
    sol     = solve_ivp(derivs_buphis, [t0, t1], V0, t_eval=t, args=(params,))
    #sol = solve_ivp(derivs, [t0, t1], V0, t_eval=t, args=(params,))
    
    print("-" * 22)
    print(f"H_pred = {sol.t[-1]:7.3f} units")

    if plot_soln:
        # Plot solution
        plt.close('all')
        
        plt.plot(sol.y.T, sol.t, '-')
        plt.legend((r'$b$', r'$\bar{u}$', r'$\phi$', r'$s$'))
        plt.grid('both')

        plt.xlabel(r'Plume parameter', size='large')
        plt.ylabel(r'Altitude, $x$', size='large')
        # plt.legend((r'$b/b_0$', r'$\bar{u}/\bar{u}_0$', r'$\phi/\phi_0$',
        #             r'$\rho_a/\rho_{a,0}$', r'$\rho_b/\rho_{b,0}$'))
        # plt.grid('both')
        plt.show()
