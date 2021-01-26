import os
import random


lowpoly_base_path = '/datasets/lowpoly_objectsV1/'


def get_lowpoly_human():

    human = 'human_lowpoly.obj'
    human_path = os.path.join(lowpoly_base_path, human)

    return human_path

def get_random_lowpoly_vehicle():

    v = random.choice(['car', 'car', 'car', 'moto', 'moto', 'bus'])
    vehicle_path = os.path.join(lowpoly_base_path, v + '.obj')

    return vehicle_path

def get_apple():
    return '/datasets/lowpoly_objectsV1/apple.obj'