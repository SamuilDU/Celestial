import numpy as np
#####

pi = np.pi
e = np.e

def runge_kutta_family(stages, time_vec, stage_vec, A_matrix, n, h, initial_conditions, function):

    t, y = np.append([initial_conditions[0], np.zeros(n - 1)]), np.append([initial_conditions[1], np.zeros(n - 1)])
    k_vec = np.zeros(len(stages))

    for k in range(n): 
        for i in range(n):
            for j in range(i):

                k_vec[i] = function(t[k] + time_vec[i] * h, y[k] + sum(np.multiply(A_matrix[i, j] * k_vec[j]))*h)

        t[k + 1] = t[k] + h
        y[k + 1] = y[k] + h * sum(np.multiply(stage_vec, k_vec))

    return t, y


def rk4_cts(param, var, h, function):

    k_1 = function(param, var) 
    k_2 = function(param + h/2, var + k_1 * h/2)
    k_3 = function(param + h/2, var + k_2 * h/2)
    k_4 = function(param + h, var + h*k_3)

    k_vec = np.array([k_1, k_2, k_3, k_4])

    return k_vec

def fractured_rk4(k, param, var, h, k_vec):

    param[k + 1] = param[k] + h # n*h = simulation time 
    var[k + 1] = var[k] + h/6 * (k_vec[0] + 2*(k_vec[1] + k_vec[2]) + k_vec[3])

    return var[k + 1], param[k + 1]


def linear(x, t):
    return x