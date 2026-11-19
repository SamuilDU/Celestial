import numpy as np
from ..general_utils import chunker, parallelization_scheme  
from itertools import repeat


def verlet(initial_time, initial_positions, initial_velocities, h, number_points, method, function, *params):
    N, M = np.shape(initial_positions) #fixes problem size (num of particles) & dimension (1D, 2D, ...)

    if method == "velocity":
        time = np.append(initial_time, np.zeros(number_points - 1))
        position = np.append(initial_positions, np.zeros((int(N/2), M * (number_points - 1)))).reshape(number_points, int(N/2) , M)
        velocity = np.append(initial_velocities[int(N/2):], np.zeros((int(N/2), M * (number_points - 1)))).reshape(number_points, int(N/2) , M)
        
        if number_points < 1e3:
            chunk_position, chunk_velocity = chunker(initial_positions, 4), chunker(initial_velocities, 4) 
            acceleration = parallelization_scheme(function, args = (repeat(initial_time, 4), chunk_position, chunk_velocity, repeat(params, 4)))
        else:
            acceleration = function(initial_time, initial_positions, initial_velocities, params) 


        while j < (number_points - 1):
            velocity[j] +=  0.5 * acceleration * h
            position[j + 1] = position[j] + velocity[j] * h
            time[j + 1] = time[j] + h 

            if number_points < 1e3:
                chunk_position, chunk_velocity = chunker(position[ + 1], 4), chunker(velocity[j + 1], 4) 
                acceleration = parallelization_scheme(function, args = (repeat(time[j + 1], N), chunk_position, chunk_velocity, repeat(params, N)))
            else:
                acceleration = function(time[j + 1], position[j + 1], velocity[j + 1], params) 

            velocity[j + 1] = velocity[j] + h * acceleration
            j += 1

        return position, velocity

    elif method == "position":
        time = np.append(initial_time, np.zeros(number_points - 1))
        position = np.append(initial_positions, np.zeros((int(N/2) * M * (number_points - 1)))).reshape(number_points, int(N/2) , M)
        
        if number_points < 1e3:
            chunk_position = chunker(initial_positions, 4) 
            acceleration = parallelization_scheme(function, args = (repeat(initial_time, 4), chunk_position, repeat(params, 4)))
        else:
            acceleration = function(initial_time, initial_positions, params) 

        position[1] = initial_positions + initial_velocities * h + 0.5 * acceleration * h**2
        j = 2
        
        while j < (number_points - 1):
            if number_points < 1e3:
                chunk_position = chunker(position[j], 4) 
                acceleration = parallelization_scheme(function, args = (repeat(time[j], 4), chunk_position, repeat(params, 4)))
            else:
                 acceleration = function(time[j], position[j], params) 

            acceleration = function(time[j], position[j], params)
            position[j + 1] = 2 * position[j] - position[j - 1] + acceleration * h**2
            j += 1
             
        return position          
