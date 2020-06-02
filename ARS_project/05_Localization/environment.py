import pygame
import operator
from copy import deepcopy

X, Y, TH = 0, 1, 2
V, O = 0, 1


class Environment:

    def __init__(self, screen, wall_color=(0, 0, 0), beacon_color=(0, 0, 0), beacon_size=10, wall_size=8,
                 margin=(10, 10)):
        self.screen = screen

        self.wall_color = wall_color
        self.wall_size = wall_size
        self.walls = []

        self.beacon_color = beacon_color
        self.beacon_size = beacon_size
        self.beacons = []
        self.inner_maze_walls = []

        self.margin_corners = [
            (margin[X], margin[Y]),
            (self.screen.get_size()[X] - margin[X], margin[Y]),
            (self.screen.get_size()[X] - margin[X], self.screen.get_size()[Y] - margin[Y]),
            (margin[X], self.screen.get_size()[Y] - margin[Y])
        ]

    def round_point(self, point):  # INVERT Y to get a right movement and axis origin
        return (int(round(point[X])), int(round(self.screen.get_size()[Y] - point[Y])))

    def sum(self, a, b):
        return tuple(map(operator.add, a, b))

    def new_sensorized_wall(self, pA, pB):
        #if not pA in self.beacons:
        #    self.beacons.append(pA)
        #if not pB in self.beacons:
        #    self.beacons.append(pB)
        if not [pA, pB] in self.walls:
            self.walls.append([pA, pB])

    def add_frame(self):
        self.new_sensorized_wall(self.margin_corners[0], self.margin_corners[1])
        self.new_sensorized_wall(self.margin_corners[1], self.margin_corners[2])
        self.new_sensorized_wall(self.margin_corners[2], self.margin_corners[3])
        self.new_sensorized_wall(self.margin_corners[3], self.margin_corners[0])

    def maze_environment(self):
        self.add_frame()
        sensor1 = self.sum(self.margin_corners[0], (0, 590))
        sensor2 = self.sum(self.margin_corners[0], (455, 590))
        sensor3 = self.sum(self.margin_corners[1], (0, 410))
        sensor4 = self.sum(self.margin_corners[1], (-455, 410))
        sensor5 = self.sum(self.margin_corners[1], (-455, 180))
        sensor6 = self.sum(self.margin_corners[0], (455, 0))
        sensor7 = self.sum(self.margin_corners[0], (455, 240))
        self.new_sensorized_wall(sensor1, sensor2)
        self.new_sensorized_wall(sensor3, sensor4)
        self.new_sensorized_wall(sensor4, sensor5)
        self.new_sensorized_wall(sensor6, sensor7)
        self.inner_maze_walls.append([sensor1, sensor2])
        self.inner_maze_walls.append([sensor3, sensor4])
        self.inner_maze_walls.append([sensor4, sensor5])
        self.inner_maze_walls.append([sensor6, sensor7])

        self.beacons.append(self.margin_corners[0])
        self.beacons.append(self.margin_corners[1])
        self.beacons.append(self.margin_corners[2])
        #self.beacons.append(self.margin_corners[3])
        #self.beacons.append(sensor1)
        self.beacons.append(sensor2)
        self.beacons.append(sensor3)
        self.beacons.append(sensor4)
        self.beacons.append(sensor5)
        self.beacons.append(sensor6)
        self.beacons.append(sensor7)




        indexed_beacons = []
        for index, beacon in enumerate(self.beacons):
            indexed_beacons.append([beacon, index])
        self.beacons = deepcopy(indexed_beacons)

    def draw_environment(self):
        for wall in self.walls:
            pygame.draw.line(self.screen, self.wall_color, self.round_point(wall[0]), self.round_point(wall[1]),
                             self.wall_size)
        for sensor in self.beacons:
            pygame.draw.circle(self.screen, self.beacon_color, self.round_point(sensor[0]), self.beacon_size,
                               self.beacon_size)
