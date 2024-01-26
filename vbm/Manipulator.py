import time
import math
import numpy as np
from threading import Thread
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Angle, Coord


class Manipulator(Thread):

    def __init__(self, queue, angle=238,dist_threshold=20) -> None:

        self.queue = queue
        self.angle = angle
        self.mycobot = MyCobot('/dev/ttyACM0')
        self.prev = np.zeros(2)
        self.threshold = dist_threshold
        self.naptime = 3

        Thread.__init__(self)

    def run(self) -> None:
        while True:
            find_coord = False
            while not self.queue.empty():
                world_coordinates_mm = self.queue.get()
                if not find_coord:
                    
                    rotated_y, rotated_x = self.transform_coordinate(world_coordinates_mm[0], (world_coordinates_mm[1]) , self.angle)
                    current = np.array([rotated_x,rotated_y])
                    dist = np.linalg.norm(current - self.prev,1)
                    if dist > self.threshold:
                        print("Move to Coordinate:",rotated_x,rotated_y," dist:",dist)
                        self.move(rotated_x,rotated_y)
                        self.prev =current
                    find_coord = True
            
            time.sleep(self.naptime)

    def transform_coordinate(self,x, y, angle_deg):
        # Convert angle from degrees to radians
        angle_rad = math.radians(angle_deg)

        # Define the rotation matrix
        rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                                    [math.sin(angle_rad), math.cos(angle_rad)]])

        # Create a column vector for the original coordinates
        original_coordinates = np.array([[x], [y]])

        # Perform the rotation transformation
        rotated_coordinates = np.dot(rotation_matrix, original_coordinates)

        # Extract the rotated x and y coordinates
        rotated_x, rotated_y = rotated_coordinates[0, 0], rotated_coordinates[1, 0]

        return rotated_x, rotated_y
    
    def move(self,rot_x,rot_y):
        coords = [int(rot_x), int(rot_y), 140, 0, 0, 0]
        self.mycobot.send_coords(coords, 70, 1)
        print("::send_coords() ==> send coords {}, speed 70, mode 0\n".format(coords))
        