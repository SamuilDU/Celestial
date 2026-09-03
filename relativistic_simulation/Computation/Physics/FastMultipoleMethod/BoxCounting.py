import numpy as np
from numpy import linalg as LA


def CanvasSize(particles_position):

    distance2origin = [LA.norm(a) for a in particles_position]
    maxdist = max(distance2origin)

    return maxdist * 1.05 # sligtly bigger than the maximum distance

         
def FourDivideSpace(center_x, center_y):
    radius = np.sqrt(2)/4 * np.sqrt(center_x**2 + center_y**2)
    return  radius * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])

def IsEmptyBox(boxCenter_x, boxCenter_y, particles_position):

    edgeRegion = 2 * FourDivideSpace(boxCenter_x, boxCenter_y)[0:2:2]

    upper_x = edgeRegion[0][0]
    lower_x = edgeRegion[1][0]
    upper_y = edgeRegion[0][1]
    lower_y = edgeRegion[1][1]

    particles_position_x = particles_position[:, 0]
    particles_position_y = particles_position[:, 1]

    val1 = sum(lower_x < particles_position_x < upper_x)
    val2 = sum(lower_y < particles_position_y < upper_y)

    if (val1 and val2) != 0:
        value = False
    else:
        value = True

    return value


