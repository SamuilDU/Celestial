import numpy as np
from numpy import linalg as LA

G = 6.67*10**(-11)



def NetForce(ivector_position, vector_masses, other_positions):

    dimensions = np.shape(other_positions)
    number_bodies = dimensions[0]

    iForce = 0

    for j in range(number_bodies):

            relative_position = ivector_position - ivector_position[j]
            iForce += -G * vector_masses[j] * relative_position / LA.norm(relative_position)**3

    return iForce