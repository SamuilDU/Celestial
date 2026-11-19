from itertools import islice, chain
from concurrent.futures import ProcessPoolExecutor
import numpy.linalg as LA
import numpy as np  

####

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


#######
def aux_cumprod(num, start, stop, step):
    cumprod = 1
    for k in range(start, stop, step):
        cumprod *= num - k

    return cumprod

#######

def parallelization_scheme(function, *args, **kwargs):

    res = []
    with ProcessPoolExecutor() as executor:
        for chunks in executor.map(function, args, kwargs):
            res.append(chunks)

    res = list(chain.from_iterable(res))
    return res


def chunker(data, num_chunks):

    if num_chunks <= 0:
        raise ValueError("Number of chunks must be a non-null, positive integer")
    else: 
        it = iter(data)
        return [list(islice(it, num_chunks)) for _ in range((len(data) + num_chunks - 1)//num_chunks)]




