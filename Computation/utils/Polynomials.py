import numpy as np
from .general_utils import aux_cumprod
from math import comb, factorial

def Legendre_coeffcients(L, m):
        coefficients = np.zeros(L + m)
        for k in range(L + 1):
            coefficients[k] = comb(L, k) * aux_cumprod(2*L - m, 0, L + m, 1)
        
        return coefficients
    
def Legendre_values(L, m, x):
        
    if m < 0:
	    m *= -1
        minus_coeff = factorial(L - m)/factorial(L + m)
        coefficients = minus_coeff * (-1)**(L + m) * Legendre_coeffcients(L, m)/(2**L * factorial(L))
        values = np.polyval(coefficients, x) * (1 - x**2)**(m/2)
    
    elif m >= 0:
        coefficients = (-1)**(L + m) * Legendre_coeffcients(L, m)/(2**L * factorial(L))
        values = np.polyval(coefficients, x)
        
    return values


def SphericalHarmonics(theta, phi, L, m, part):
    #Condon-Shortley phase already included inside the Legendre Polynomials

    factor = np.sqrt((2*L + 1)*factorial(L - m)/(factorial(L + m) * 4 * np.pi))

    if part == "real":
        phase = np.cos(phi)
    elif part == "Imaginary":
        phase = np.sin(phi)

    sph_har = factor * np.multiply(Legendre_values(np.cos(theta), L, m), phase)

    return sph_har