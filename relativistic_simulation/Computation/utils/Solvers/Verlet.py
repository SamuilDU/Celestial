import numpy as np
from ..general_utils import chunker, parallelization_scheme  

def verlet(initial_time, initial_positions, initial_velocities, h, number_points, method, function, *params):
    N, M = np.shape(initial_positions) #fixes problem size (num of particles) & dimension (1D, 2D, ...)

    if method == "velocity":
        time = np.append(initial_time, np.zeros(number_points - 1))
        position = np.append(initial_positions, np.zeros((int(N/2), M * (number_points - 1)))).reshape(number_points, int(N/2) , M)
        velocity = np.append(initial_velocities[int(N/2):], np.zeros((int(N/2), M * (number_points - 1)))).reshape(number_points, int(N/2) , M)


        if number_points >= 1e3:
            chunks_positions, chunks_velocities = chunker(initial_positions, 4), chunker(initial_velocities, 4)
            acceleration = parallelization_scheme(function, args=(initial_time, chunks_positions, chunks_velocities), kwargs= None)

        else: 
            acceleration = function(initial_time, initial_positions, initial_velocities, params)


        while j < (number_points - 1):
            velocity[j] +=  0.5 * acceleration * h
            position[j + 1] = position[j] + velocity[j] * h
            time[j + 1] = time[j] + h 

            acceleration = function(time[j + 1], position[j + 1], velocity[j], params)
            velocity[j + 1] = velocity[j] + h * acceleration
            j += 1

        return position, velocity

    elif method == "position":
        time = np.append(initial_time, np.zeros(number_points - 1))
        position = np.append(initial_positions, np.zeros((int(N/2) * M * (number_points - 1)))).reshape(number_points, int(N/2) , M)
        acceleration = function(initial_time, initial_positions, params)

        position[1] = initial_positions + initial_velocities * h + 0.5 * acceleration * h**2
        j = 2
        
        while j < (number_points - 1):
            acceleration = function(time[j], position[j], params)
            position[j + 1] = 2 * position[j] - position[j - 1] + acceleration * h**2
            j += 1
             
        return position          
