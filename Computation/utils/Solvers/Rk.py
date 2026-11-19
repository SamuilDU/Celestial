import numpy as np

rk_family = dict({

    "explicit":{
        "first":{
            "Euler":{
                "stage_vec" : np.array([1]),
                "time_vec" : np.array([0]),
                "Runge_matrix" : np.array([0])
            } 
        },

        "second":{
            "Midpoint":{
                "stage_vec" : np.array([0, 1]),
                "time_vec" : np.array([0, 1/2]),
                "Runge_matrix" : np.array([[0, 0], [1/2, 0]])
            },
            "Heun":{
                "stage_vec" : np.array([1/2, 1/2]),
                "time_vec" : np.array([0, 1]),
                "Runge_matrix" : np.array([[0, 0], [1, 0]])
            },
            "Ralston":{
                "stage_vec" : np.array([1/4, 3/4]),
                "time_vec" : np.array([0, 1]),
                "Runge_matrix" : np.array([[0, 0], [1, 0]])
            }
        },

        "third":{
            "Kutta":{
                "stage_vec" : np.array([1/6, 2/3, 1/6]),
                "time_vec" : np.array([0, 1/2, 1]),
                "Runge_matrix" : np.array([[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]])
            },
            "Heun":{
                "stage_vec" : np.array([1/4, 0, 3/4]),
                "time_vec" : np.array([0, 1/3, 2/3]),
                "Runge_matrix" : np.array([[0, 0, 0], [1/3, 0, 0], [0, 2/3, 0]])
            },
            "Ralston":{
                "stage_vec" : np.array([2/9, 1/3, 4/9]),
                "time_vec" : np.array([0, 1/2, 3/4]),
                "Runge_matrix" : np.array([[0, 0, 0], [1/2, 0, 0], [0, 3/4, 0]])
            },
            "SSPRK3":{
                "stage_vec" : np.array([1/6, 1/6, 2/3]),
                "time_vec" : np.array([0, 1, 1/2]),
                "Runge_matrix" : np.array([[0, 0, 0], [1, 0, 0], [1/4, 1/4, 0]])
            }
        },

        "fourth" :{
            "RK4":{
                "stage_vec" : np.array([1/6, 1/3, 1/3, 1/6]),
                "time_vec" : np.array([0, 1/2, 1/2, 1]),
                "Runge_matrix" : np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]])
            },
            "3/8":{
                "stage_vec" : np.array([1/8, 3/8, 3/8, 1/8]),
                "time_vec" : np.array([0, 1/2, 1/2, 1]),
                "Runge_matrix" : np.array([[0, 0, 0, 0], [1/3, 0, 0, 0], [-1/3, 1, 0, 0], [1, -1, 1, 0]])
            }
    }
}})



def runge_kutta_family(method, order, type, number_points, step, initial_conditions, function):

    stage_vec, time_vec, Runge_matrix = rk_family[method][order][type].values()
    stages = len(stage_vec)
    
    t, y = np.append(initial_conditions[0], np.zeros(number_points - 1)), np.append(initial_conditions[1], np.zeros(number_points - 1))
    k_vec = np.zeros(stages)

    for k in range(number_points - 1): 
        for s in range(stages - 1):
            r = sum(np.multiply(Runge_matrix[s, :], k_vec[s]))
            k_vec[s + 1] = function(t[k] + time_vec[s] * step, y[k] + r*step)

        t[k + 1] = t[k] + step
        y[k + 1] = y[k] + step * sum(np.multiply(stage_vec, k_vec))

    return t, y




