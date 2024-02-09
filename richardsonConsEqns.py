#!/usr/bin/python3

""" 
richardsonConsEqns.py 
"""

import numpy as np

from scipy.integrate import ode
from OneDPlumeModels.utils import integrator


def derivs(x, V, params=(.1, 1)):
    """
    Describes the plume model that "conserves" Richardson number

    Parameters
    ----------
    x : int
        altitude within the plume
    V : array_like
        State variables, V = [Q, M, Ri]
    params : tuple or array_like
        Additional parameters, entrainment coefficient and atmospheric 
        stratification params = (alpha, N2)

    Returns
    -------
    dVdx : array_like
        array of derivatives of the state vector

    Changes log
    -----------
    2019-11-21 : Created by D. E. Jessop, OVSG-IPGP/OPGC-LMV
    2023-04-18 : (DEJ) minor edits
    2023-11-15 : (DEJ) corrections
    2024-01-29 : (DEJ) minor edits to how params are passed to fns 
    """
    Q, M, Ri = V
    alpha, N2 = params
    
    M3_2  = np.power(M, 1.5)
    M5_2  = M3_2 * M

    dVdx    = np.zeros_like(V)
    dVdx[0] = 2 * alpha * np.sqrt(M)
    dVdx[1] = Ri * M3_2 / Q
    dVdx[2] = (-N2 * Q**3 / M5_2 - 2.5 * Ri / M * dVdx[1]
               + 2 * Ri / Q * dVdx[0])

    return dVdx


def jac(x, V, params=(.1, 1)):
    """
    Jacobian of derivatives for the plume model that "conserves" Richardson
    number

    Parameters
    ----------
    x : int
        altitude within the plume
    V : array_like
        State variables, V = [Q, M, Ri]
    params : tuple or array_like
        Additional parameters, entrainment coefficient and atmospheric 
        stratification params = (alpha, N2)

    Returns
    -------
    jac : array_like
        jacobian matrix for the derivatives

    Changes log
    -----------
    2023-04-24: Created by D. E. Jessop - to be finished.

    """
    Q, M, Ri = V
    alpha, N2 = params
    
    M1_2  = np.power(M, 1.5)
    M3_2  = M1_2 * M
    M5_2  = M3_2 * M
    M7_2  = M5_2 * M
    #dVdx  = derivs(x, V, params)

    a = -3  * N2 * Q**2 / M5_2 - (4 * alpha - 2.5  * Ri) * Ri * M1_2 / Q**2
    b = 2.5 * N2 * Q**3 / M7_2 + (2 * alpha - 1.25 * Ri) * Ri / (Q * M1_2)
    c = (4 * alpha - 5 * Ri) * M1_2 / Q
    #dVdx[2] = -N2 * Q / M5_2 - 2.5 * Ri * M * dVdx[1] + Ri / Q * dVdx[0]

    jac       = np.zeros(np.array([1, 1]) * len(V))
    jac[0, 0] = 0
    jac[0, 1] = alpha / np.sqrt(M)
    jac[0, 2] = 0
    jac[1, 0] =      -Ri * M3_2 / Q**2
    jac[1, 1] = 1.5 * Ri * M1_2 / Q
    jac[1, 2] =            M3_2 / Q
    jac[2, 0] = a
    jac[2, 1] = b
    jac[2, 2] = c

    return jac



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x0 =  0.
    b0 =   .5
    u0 =  50
    g0 = -10
    V0 = [b0**2 * u0,
          b0 **2 * u0**2,
          g0 * b0 / u0**2]
    p  = (.1, 1)
    
    x, V = integrator(p, x0, V0, derivs, jac=jac, t1=20)

    q, m, Ri = V.T
    b, u     = q / np.sqrt(m), m / q
    Vp       = np.vstack([b, u, np.abs(Ri)]).T

    plt.close('all')
    plt.plot(Vp, x, '-')
    plt.legend(('b', 'u', '|Ri|'))
    plt.xscale('log')
    plt.xlabel('Plume parameter')
    plt.ylabel(r'Altitude, $x$')
    plt.show()
