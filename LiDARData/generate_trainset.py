import random
import matplotlib.pyplot as plt
import math
import json

TRAINSET = 'trainset.json'

def polar_to_cartesiane(angle, distance):
    return (distance * math.cos(angle), distance * math.sin(angle))

def generate_LiDAR(num=360, max_distance=12.0):
    """
    This function only generates fake LiDAR points for simulation.

    params: 
    - num (int): number of LiDAR points.
    - max_distance (float): maximum distance for LiDAR measurements.

    return: list of tuple (angle, distance). Might be a dictionnary I still don't know.
    """

    lidar_data = []
    for angle in range(0, 360, 360 // num):
        distance = random.uniform(10, max_distance)
        lidar_data.append({'angle': angle, 'distance': distance})

    random_points = random.randint(0, num-1)
    random_range = random.randint(5, 20)
    random_distance = random.randint(2, 11)

    for i in range(random_range):
        if (random_points + i) < num:
            lidar_data[random_points + i]['distance'] = random_distance

    lidar_data[90]['distance'] = 8
    lidar_data[0]['distance'] = 8
    lidar_data[180]['distance'] = 8
    lidar_data[-90]['distance'] = 8

    return lidar_data

def LiDAR_to_trainset(lidar_data: list):
    with open(TRAINSET, 'w'):
        for data in lidar_data: 
            (x, y) = polar_to_cartesiane(data['angle'], data['distance'])
            print(x, y)

lidar_data = generate_LiDAR()
#print(lidar_data)
print(lidar_data)

cartesian_data = [(d['distance'] * math.cos(math.radians(d['angle'])), d['distance'] * math.sin(math.radians(d['angle']))) for d in lidar_data]

x_coords, y_coords = zip(*cartesian_data)

plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, marker='o', label='LiDAR Points')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('LiDAR Data Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()