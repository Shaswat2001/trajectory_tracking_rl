import math
import numpy as np

def generate_sample_trajectory(q_start,q_des):

    fraction = np.linspace(0,1,10)
    points = q_start + fraction[:,np.newaxis]*(q_des - q_start)
    return np.round(points,3)

trajectory1 = generate_sample_trajectory(np.array([0,0,2]),np.array([1,1,2]))
trajectory2 = generate_sample_trajectory(np.array([1,1,2]),np.array([2,0,2]))
print(np.concatenate((trajectory1[:-1,:],trajectory2)))
# print(np.concatenate((trajectory1,trajectory2)))

# def get_subregion():
#         global trajectory

#         points = np.zeros((3,3))
#         points_list = trajectory[98:min(98+3,99+1),:]
#         points[:len(points_list)] = points_list
#         points[len(points_list):,:] = points[len(points_list)-1]
#         points -= np.array([1,1,1])

#         return points.flatten()

# print(get_subregion())