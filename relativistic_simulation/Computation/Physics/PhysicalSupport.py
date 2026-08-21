
import numpy as np


G = 6.67 * 10**(-11)
c = 299792458
h = 6.62607015*10**(-34)  
q = 1.602176634*10**(-19)
k_B = 1.380649*10**(-23)

##

def inverse_rad(theta, theta_0, eccent, mu, M, L_0): # solution to Binet's equation, theta=theta_0 determines r_0^-1
    return (G*M*mu/L_0**2) * (1 + eccent * np.cos(theta - theta_0))

def accel(M, r, t):
    G = 6.67 * 10**(-11)
    return -G*M/r**2

def mod_angular_momentum(mu, r_0, v_0, alpha):
    return mu*r_0*v_0*np.abs(np.sin(alpha))

def eccentricity(r_0, v_0, alpha, mu, M ):

    E_0, L_0 = 1/2*mu*v_0**2 - G*M*mu/r_0, mod_angular_momentum(mu, r_0, v_0, alpha)

    return np.sqrt(1 + 2*E_0*L_0**2 / ((G*M)**2)*mu)


def radial_vel(r, eccent, mu, M, L_0):
    return (L_0**3 *eccent/(G*mu**2 * M)) * np.sqrt( ( 1 + eccent**2)*r**2 - 2*(L_0/(G*M*mu))*r + (L_0**2/(G*M*mu))**2)

