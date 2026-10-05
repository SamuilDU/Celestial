import numpy as np
from numpy import linalg as LA
from itertools import islice
from concurrent.futures import ProcessPoolExecutor


G = 6.67*10**(-11)

def chunker(data, num_chunks):

    if num_chunks <= 0:
        raise ValueError("Number of chunks must be a non-null, positive integer")
    else: 
        it = iter(data)
        return [list(islice(it, num_chunks)) for _ in range((len(data) + num_chunks - 1)//num_chunks)]


def NetForces(positions, masses):

      n = len(masses)
      Forces = np.zeros(np.shape(positions))

      for i in range(n):
            imass = masses[i]
            iposition = positions[i]

            for jposition, jmass in zip(positions, masses):
                  distance = LA.norm(iposition - jposition)
            
                  if distance > 10**4: #reescale correctly
                        Forces[i] += -G * (imass * jmass / distance**2) * (iposition - jposition)/distance 

      return Forces