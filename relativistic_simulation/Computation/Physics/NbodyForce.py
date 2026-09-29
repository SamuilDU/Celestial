import numpy as np
from numpy import linalg as LA


G = 6.67*10**(-11)

#def NetForces(positions, masses):
#
#      n = len(masses)
#      Forces = np.zeros(np.shape(positions))
#
#      for i in range(n):
#            imass = masses[i]
#            iposition = positions[i]
#
#            for jposition, jmass in zip(positions, masses):
#                  distance = LA.norm(iposition - jposition)
#            
#                  if distance > 10**4: #reescale correctly
#                        Forces[i] += -G * (imass * jmass / distance**2) * (iposition - jposition)/distance 
#
#      return Forces

def NetForce(ivector_position, imass, other_masses, other_positions):

      for position, mass in zip(other_positions, other_masses):

            distance = LA.norm(ivector_position - position)
            acceleration = np.zeros(np.shape(ivector_position))
            
            if distance > 10**4:
                 acceleration += (mass / distance**2) * position/distance #Rescale the problem correctly!!! 

      return - G *imass * acceleration
 