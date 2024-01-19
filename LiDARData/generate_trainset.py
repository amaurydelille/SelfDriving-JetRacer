import random
import matplotlib.pyplot as plt
import math
import json

LEFT = 0
RIGHT = 1
FORWARD = 2
STOP = 3
TRAINSET = 'LiDARData/trainset.json'

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

    label = STOP
    if random_points > 0 and random_points < 90:
        label = LEFT
    if random_points < 0 and random_points > -90:
        label = FORWARD
    if random_points >= -90 and random_points < 180:
        label = FORWARD
    if random_points >= 90 and random_points < 180:
        label = RIGHT

    return (lidar_data, label)

def LiDAR_to_trainset(lidar: list):
    trainset = []
    for data in lidar[0]:
        (x, y) = polar_to_cartesiane(math.radians(data['angle']), data['distance'])
        trainset.append({'x': x, 'y': y})

    trainset.append({'label': lidar[1]})

    with open(TRAINSET, 'w') as json_file:
        json.dump(trainset, json_file, indent=1)

(lidar_data, label) = generate_LiDAR()
print(lidar_data)
LiDAR_to_trainset((lidar_data, label))

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