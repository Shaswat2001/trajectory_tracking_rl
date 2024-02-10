import sys
sys.path.insert(1, "/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV/")

import numpy as np
import yaml

with open("/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV/Config/obstacle_1.yaml", "r") as stream:
    try:
        obstacles = yaml.safe_load(stream)["obstacle_coordinates"]
    except yaml.YAMLError as exc:
        print(exc)

coordinates = []
for key,coordinate in obstacles.items():

    coordinates.append(coordinate)

print(coordinates[0])

sample_position = [[2,2,2],[4,3,3],[1,2,2],[-1,-2,-3],[-5,-3,-4],[0,0,0],[-1.5,-1.5,-1.5]]
for coordinate in coordinates:
    min_crd = coordinate - np.array([1,1,1])/2
    max_crd = coordinate + np.array([1,1,1])/2
    print(f"min coordinate is : {min_crd}")
    print(f"max coordinate is : {max_crd}")

for position in sample_position:
    for coordinate in coordinates:
        min_crd = coordinate - np.array([1,1,1])/2
        max_crd = coordinate + np.array([1,1,1])/2
        if np.all(position > min_crd) and np.all(position < max_crd):
            print("WITHIN OBSTACLE")
        else:
            print("OUTSIDE OBSTACLE")
