import math
import numpy as np
import matplotlib.pyplot as plt
def generate_sample_trajectory(q_start,q_des,curve_type="line"):

    if curve_type == "line":
        fraction = np.linspace(0,1,10)
        points = q_start + fraction[:,np.newaxis]*(q_des - q_start)
        return np.round(points,3)
    elif curve_type == "sin":

        if q_start[0] == q_des[0]:

            fraction = np.linspace(0,1,100)
            print(np.sin(2*np.pi*fraction))
            points = q_start + np.sin(2*np.pi*fraction)[:,np.newaxis]*(q_des - q_start)

        else:
            slope = (q_start[1] - q_des[1])/(q_start[0] - q_des[0])
        
            theta= math.atan(slope)
            cos = math.cos(theta)
            sin = math.sin(theta)

        return points

trajectory1 = generate_sample_trajectory(np.array([0,0,2]),np.array([1,0,2]),curve_type="sin")
print(trajectory1)
plt.plot(trajectory1[:,0],trajectory1[:,1])
plt.show()
# trajectory2 = generate_sample_trajectory(np.array([1,1,2]),np.array([2,0,2]))
# print(np.concatenate((trajectory1[:-1,:],trajectory2)))
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