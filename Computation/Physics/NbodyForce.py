import numpy as np
from numpy import linalg as LA

G = 6.67*10**(-11)

def NetAcceleration(t, positions, masses):

      n = len(masses)
      acceleration = np.zeros(np.shape(positions))

      for i in range(n):
            
            iposition = positions[i]

            for jposition, jmass in zip(positions, masses):
                  distance = LA.norm(iposition - jposition)
            
                  if distance > 10**4: #reescale correctly
                        acceleration[i] += -G * (jmass / distance**2) * (iposition - jposition)/distance 

      return acceleration