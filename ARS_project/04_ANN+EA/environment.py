
import pygame

L, R = 0, 1
X, Y, TH = 0, 1, 2
class Environment():

	def __init__(self, screen, color, walls, size = 8):
		self.screen = screen
		self.color = color
		self.walls = walls
		self.size = size

	def round_Y(self, point):  # INVERT Y to get a right movement and axis origin
		return (int(round(point[X])), int(round(self.screen.get_size()[Y] - point[Y])))

	def draw_environment(self):
		for wall in self.walls:
			pygame.draw.line(self.screen, self.color, self.round_Y(wall[0]), self.round_Y(wall[1]), self.size)

