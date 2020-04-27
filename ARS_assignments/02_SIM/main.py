
import pygame
import sys
import numpy as np
import math
from pygame.locals import K_t, K_g, K_w, K_s, K_o, K_l, K_x, KEYDOWN
import time
import robot as rb

#######################################################
################# PROPERTIES ##########################
SIZE_SCREEN = width, height = 1000, 700
COLOR_SCREEN = 255,255,255
COLOR_ENVIROMENT = 90, 90, 255
MAX_DISTANCE_SENSOR = 200
MAX_VELOCITY = 100
MOTOR_GRIP = MAX_VELOCITY/10
ROBOT_RADIUS = 30
ROBOT_POSITION = [200, 300, 60] # X Y THETA in GRAD
DELTA_T = .03
FPS = 200 #Frames per second
#######################################################
#######################################################
L, R = 0, 1
X, Y, TH = 0, 1, 2

class Environment():
	def __init__(self, walls):
		self.walls = walls

	def round_Y(self, point):  # INVERT Y to get a right movement and axis origin
		return (int(round(point[X])), int(round(screen.get_size()[Y] - point[Y])))

	def draw_environment(self):
		#rects = []
		for wall in self.walls:
			pygame.draw.line(screen, COLOR_ENVIROMENT, self.round_Y(wall[0]), self.round_Y(wall[1]), 4)
			#rects.append()
		#return rects

# == MAIN ==
collision_flag = False # Inidcator of a collision
pygame.init()  # Initializing library
screen = pygame.display.set_mode(SIZE_SCREEN)  # Initializing screen
FPSCLOCK = pygame.time.Clock()  # Refreshing screen rate

#init environment
env = Environment(
	[
		[  (30,30), (970,30)],
		[ (970,30),(970,670)],
		[(970,670), (30,670)],
		[ (30,670),  (30,30)],
		[(200,200),(800,200)],
		[(200,200),(500,500)],
		[(800,200),(500,500)],
	]
)

#limits_environment = env.get_limits() # Getting the boundaries of the environment
env.draw_environment() # Drawing the environment

#init robot
robot = rb.Robot(2*ROBOT_RADIUS, MAX_VELOCITY, MAX_DISTANCE_SENSOR)
robot.position = [ROBOT_POSITION[0], ROBOT_POSITION[1], math.radians(ROBOT_POSITION[2])]
robot.robot_moving(screen, env.walls, DELTA_T)

# Main loop of the game
while True:
	screen.fill(COLOR_SCREEN) # Background screen

	for event in pygame.event.get():  # Event observer
		if event.type == pygame.QUIT: # Exit
			pygame.quit()
			sys.exit(1)

		##### >>>>> COMPLEX MOVE <<<<< #####
		if event.type == KEYDOWN: # Press key
			if event.key == K_w:
				robot.ChangeMotorVelocity(L,  MOTOR_GRIP)
			if event.key == K_s:
				robot.ChangeMotorVelocity(L, -MOTOR_GRIP)
			if event.key == K_o:
				robot.ChangeMotorVelocity(R,  MOTOR_GRIP)
			if event.key == K_l:
				robot.ChangeMotorVelocity(R, -MOTOR_GRIP)
			if event.key == K_t:
				robot.ChangeMotorVelocity(L,  MOTOR_GRIP)
				robot.ChangeMotorVelocity(R,  MOTOR_GRIP)
			if event.key == K_g:
				robot.ChangeMotorVelocity(L,  -MOTOR_GRIP)
				robot.ChangeMotorVelocity(R,  -MOTOR_GRIP)
			if event.key == K_x:
				robot.NewMotorVelocity(L,  0)
				robot.NewMotorVelocity(R,  0)

	#Update robot and environment
	env.draw_environment() # Drawing the environment
	robot.robot_moving(screen, env.walls, DELTA_T)

	#Update screen
	pygame.display.update()
	FPSCLOCK.tick(FPS)
