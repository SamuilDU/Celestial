import numpy as np
from .general_utils import aux_cumprod
from math import comb, factorial



def LegendrePolynomial(x, l, m):
    polynomial = 0

    if np.abs(m) > l:
        raise ValueError("The value of m must be between l and -l")

    if m < 0:
        m *= -1
        for k in range(l + 1):
           exponent = 2*l - m + k
           if exponent < 0:
               break       
           polynomial += (-1)**(l + m) * comb(l, k) * aux_cumprod(2*l - m, 0, l + m + 1, 1) * x**(exponent)
    
        polynomial *= (1 - x**2)**(m/2)/(factorial(l) * 2**l)
        polynomial *= (-1)**m *factorial(l - m)/factorial(l + m)

    else:
        for k in range(l + 1):
           exponent = 2*l - m + k
           if exponent < 0:
               break       
           polynomial += (-1)**(l + m) * comb(l, k) * aux_cumprod(2*l - m, 0, l + m + 1, 1) * x**(exponent)
    
        polynomial *= (1 - x**2)**(m/2)/(factorial(l) * 2**l)

    return polynomial

def SphericalHarmonics(theta, phi, l, m, part):
    #Condon-Shortley phase already included inside the Legendre Polynomials
    #Convention: Magnetic, Geodesic, Quantum/Base

    factor = np.sqrt((2*l + 1)*factorial(l - m)/(factorial(l + m) * 4 * np.pi))

    if part == "real":
        phase = np.cos(phi)
    elif part == "Imaginary":
        phase = np.sin(phi)

    sph_har = factor * LegendrePolynomial(np.cos(theta), l, m) * phase

    return sph_har