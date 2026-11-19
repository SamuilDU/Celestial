import numpy as np
import numpy.linalg as LA


def normal_decomposition(vector, axis):

    #correctly normalizes the axis in case it was not
    if np.abs(LA.norm(axis) - 1.01) > 1e-2 :
        axis = axis/LA.norm(axis)
    
    radial_vector = np.dot(vector, axis) * axis
    normal_vector = vector - axis
    
    return radial_vector, normal_vector
    
def collision_scheme(mass1, mass2, momentum10, momentum20):
    #free ellastic collision
    a = mass1/mass2
    g1 = (a * momentum20 - momentum10 + a*(momentum10 + momentum20))/(1 + a)
    g2 = (momentum10 - a * momentum20 + (momentum20 + momentum10))/(1 + a)
    
    return g1, g2
