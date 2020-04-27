import pygame
import shapely
from shapely.geometry import Point
from shapely import affinity
from copy import deepcopy

X, Y, TH = 0, 1, 2
class Dust:
    def __init__(self, screen, dust_size):
        self.dust_size = dust_size
        self.screen = screen
        self.dust_list = self.initialize_dust() # [[(x,y), isCleaned],[(x,y), isCleaned],[(x,y), isCleaned]]
        # self.initial_position = # dust point posizioni in range

    def initialize_dust(self):
        dusts = []
        init_margin = 25
        line_dust = int(self.dust_size / 10)
        dust_init_x = init_margin
        dust_init_y = init_margin
        dust_finish_x = self.screen.get_size()[X]
        dust_finish_y = self.screen.get_size()[Y]

        step_x = (dust_finish_x - dust_init_x) / (line_dust)
        step_y = (dust_finish_y - dust_init_y) / (line_dust)
        for i in range(line_dust):
            x_dust = dust_init_x + i*step_x
            for j in range(line_dust):
                y_dust = dust_init_y + j*step_y
                dusts.append([( x_dust, y_dust ), False])
        return dusts

    def round(self, value):
        return int(round(value))

    def round_Y(self, value):  # INVERT Y to get a right movement and axis origin
        return int(round(self.screen.get_size()[Y] - value))

    def draw_dust(self):
        for idx, dust in enumerate(self.dust_list):
            # print(dust[1], idx)
            color = (0, 255, 100)
            if dust[1] == True:
                color = (90, 90, 255)
            x = self.round(dust[0][X])
            y = self.round_Y(dust[0][Y])
            pygame.draw.rect(self.screen, color, (x, y, 5, 5))

    def update_dust(self, robot):
        robot_center = Point(robot.position[X], robot.position[Y]).buffer(1)
        robot_radius = int(0.5*robot.axis_length)
        robot_shape = shapely.affinity.scale(robot_center, robot_radius, robot_radius)
        dust_array = deepcopy(self.dust_list)
        for idx, dustx in enumerate(dust_array):
            point = Point(dustx[0][0], dustx[0][1])

            if not dustx[1]:
                if point.within(robot_shape):
                    self.reached(idx)
        self.draw_dust()

    def reached(self,index):
        # print (self.dusts[75][1])
        # print (index)
        self.dust_list[index][1] = True

    def get_score_dust(self):
        dust_array = deepcopy(self.dust_list)
        score = 0
        for dust in dust_array:
            if dust[1]:
                score += 1
        return score