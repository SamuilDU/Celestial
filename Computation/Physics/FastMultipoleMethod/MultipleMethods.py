import numpy as np
from math import comb

def LocalCoefficients(position_particles, weight_particles):

    m = len(weight_particles)

    a_coefficients = np.zeros(m)
    for k in range(1, m + 1, 1):
        a_coefficients[k] = - 1/k * weight_particles * position_particles**k #*, ** == element-wise

    a_coefficients = np.append([sum(weight_particles), a_coefficients])

    return a_coefficients

def ShiftingCoefficents(center, localCoefficients, truncation_order):

    b_coeffcients = np.zeros(truncation_order)

    for l in range(1, truncation_order + 1, 1):
        b_coeffcients[l] = sum([localCoefficients[k] * center**(l - k) * comb(l, k) - (localCoefficients[0] * center**l)/l] for k in range(1, l +1, 1))

    return b_coeffcients


def EvaluatingCoefficients(center, a_coefficients, truncation_order):

    c_coefficients = np.zeros(len(truncation_order))

    for k in range(1, len(a_coefficients) - 1, 1):
        c_coefficients[0] += a_coefficients[0]*np.log(-center) + (-1)**k * a_coefficients**k / center**k

    for l in range(1, truncation_order + 1, 1):
        c_coefficients[l] = 1/center**l * (sum(a_coefficients[k]/center**k * comb(l + k -1, k - 1)) for k in range(l)) -a_coefficients[0]/(l * center**l) #Revise th generators

    return c_coefficients


def TruncationOrder(particle_weights, distance, radius, center, desired_error, method):
	
	error = 1
	truncation_order = 1
	A = sum(particle_weights)

	while error > desired_error:

		c = distance/radius

		if method == "Multipole":
			error = A/(c - 1) * (1/c)**(truncation_order)
			
		if method == "Local":
			relative_distance = (center + radius)/distance
			error = relative_distance**(truncation_order + 1) * A/(1 - relative_distance)
			
		if method == "Multiple2Local":	
			error = (1/c)**(truncation_order + 1)*A*(c**2 + 4*np.e*(c+1)*(c + truncation_order))/(c*(c - 1))

		truncation_order += 1	

	return truncation_order


#Here all the coefficients must be previously added at a specific tree level
#Particle weights and positions must be obtained for a specific box at tree level, so as to paralelize the process
# Finally all potentials must be summed at the desired point
def MultipleMethods(dimension, method, distance , a_coefficients, b_coefficients, c_coefficients, truncation_order): 
	if dimension == 2:
		if method == "Multipole":
			potential = a_coefficients[0] * np.log(distance) + sum((a_coefficients[k]/distance**k) for k in range(truncation_order)) # Revise the generators!!!!!!
    
		elif method == "Local":
			potential = a_coefficients[0] * np.log(distance) + sum((b_coefficients / distance**l) for l in truncation_order)
	

#	else:
#		if method == "Multipole":
#			paste functions
#		if method == "Local":
#			paste functions
#		if method == "Evaluation":
#			paste functions
 
	
	return potential 