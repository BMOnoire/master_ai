import pygame
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely import affinity
import math as m

pygame.init
pygame.font.init()

#Parameterrobot
#robotVariables = np.array([3,1]) #robotVariables[x,y,direction]
#visualisation parameters
fps = 100
frametime = int(round(1000/fps))
radius = 25
myfont = pygame.font.SysFont('', 16)

#sensor parameters

NUM_OF_SENSORS = 12  # nr of sensors
SENSOR_LINE_LIMIT = 200
SENSOR_LINES = []
WALLS = []
SENSOR_READING = np.zeros(NUM_OF_SENSORS)

#Motion parameters
length_of_robot = radius*2
half_length = radius
time_step = frametime
velocity_right = 0
velocity_left = 0
size_space = 20
speed_increment = 0.1

#Methods visualization

def initialiseWindow():
    global win
    win = pygame.display.set_mode((702, 702))
    pygame.display.set_caption("Visuals")
    initializeEnvironment()
    return

def initializeEnvironment():
    for line in WALLS:
        pygame.draw.line(win, (20,20,20), line.coords[0], line.coords[1], 2)

def keypresses(x,y,direction):
    global velocity_right, velocity_left
    keys = pygame.key.get_pressed()
    position = [x, y, direction]
    #Here you can use keypresses to update the different velocities of the different wheels
    if keys[pygame.K_w]:
        velocity_left = velocity_left + speed_increment
    if keys[pygame.K_s]:
        velocity_left = velocity_left - speed_increment
    if keys[pygame.K_o]:
        velocity_right = velocity_right + speed_increment
    if keys[pygame.K_l]:
        velocity_right = velocity_right - speed_increment
    if keys[pygame.K_t]:
        velocity_left = velocity_left + speed_increment
        velocity_right = velocity_right + speed_increment
    if keys[pygame.K_g]:
        velocity_left = velocity_left - speed_increment
        velocity_right = velocity_right - speed_increment
    if keys[pygame.K_x]:
        velocity_right = 0
        velocity_left = 0

    #Here you'd use the motion method when properly integrated
    position = motion(position)
    x = position[0]
    y = position[1]
    direction = position[2]
    return x, y, direction

def createwalls():
    # Wall Line 1
    C = (1, 0)
    D = (700, 0)
    wall_line1 = LineString([C, D])
    WALLS.append(wall_line1)

    # Wall Line 2
    C = (700, 0)
    D = (700, 700)
    wall_line2 = LineString([C, D])
    WALLS.append(wall_line2)

    # Wall Line 3
    C = (700, 700)
    D = (1, 700)
    wall_line3 = LineString([C, D])
    WALLS.append(wall_line3)

    # Wall Line 4
    C = (1, 700)
    D = (1, 0)
    wall_line4 = LineString([C, D])
    WALLS.append(wall_line4)

    # Wall Line 5 (for collision testing)
    C = (500, 700)
    D = (500, 0)
    wall_line4 = LineString([C, D])
    WALLS.append(wall_line4)

# Methods motion


def rotation_matrix():
    wdt = calculate_angle()
    rot_mat = np.array([[m.cos(wdt), -m.sin(wdt), 0],
               [m.sin(wdt), m.cos(wdt), 0],
               [0, 0, 1]])
    return rot_mat

def calculate_angle():
    return (velocity_right - velocity_left)/length_of_robot

def calculate_r():
    return half_length * ((velocity_left + velocity_right)/(velocity_right - velocity_left))

def calculate_icc(position):
    r = calculate_r()
    return [(position[0] - r*m.sin(position[2])), (position[1] + r*m.cos(position[2]))]

#calculate the new position from the old position using the formula from the lecture
def motion (position):

    #special case for when the robot moves straight ahead
    if velocity_right == velocity_left:
        angle = position[2]
        x_change = m.cos(angle)*velocity_right
        y_change = m.sin(angle)*velocity_right
        position[0] = position[0]+x_change
        position[1] = position[1]+y_change
        return position

    rot_mat = rotation_matrix()
    icc = calculate_icc(position)
    wdt = calculate_angle()#*time_step
    translate = [(position[0]-icc[0]), (position[1]-icc[1]), position[2]]
    icc.append(wdt)
    new_position = rot_mat.dot(translate)+icc
    if (new_position[2] > 2*m.pi):
        new_position[2] = new_position[2] % (2 * m.pi)
    return new_position

#Checks for colision by checking whether the body of the robot or the traveled line of the robot intsects with any walls
def check_collision(x,y,direction, old):
    circle = Point(x, y).buffer(1)
    ellipse = shapely.affinity.scale(circle, radius, radius)
    traveled_line = LineString([(x, y), (old[0], old[1])])
    lines = []
    for line in WALLS:
        if ellipse.intersects(line) or traveled_line.intersects(line):
            #If the robot intersects with more than one wall, it's in a corner and should stay there
            if len(lines) > 0:
                return old[0], old[1], direction
            new = [x,y,direction]
            velocity_vector = np.subtract(new,old)
            wall_vector = np.subtract(line.coords[0], line.coords[1])
            #If the wall and the velocity are not orthogonal, find the component of the velocity along the wall
            if wall_vector.dot([velocity_vector[0], velocity_vector[1]]) != 0:
                new_vel = along_wall([velocity_vector[0], velocity_vector[1]], wall_vector)
                new_vel = np.append(new_vel,0)
                new = np.add(old, new_vel)
                x = new[0]
                y = new[1]
            #If they are orthogonal, put the robot as close as possible to the wall
            else:
                intersection_point = (0, 0)
                if ellipse.intersects(line):
                    intersect_line = ellipse.intersection(line)
                else:
                    intersect_line = traveled_line.intersection(line)
                if len(intersect_line.coords) > 1:
                    first_point = intersect_line.coords[0]
                    second_point = intersect_line.coords[1]
                    point_x = (first_point[0] + second_point[0])/2
                    point_y = (first_point[1] + second_point[1])/2
                    intersection_point = (point_x, point_y)
                elif intersect_line.geom_type == 'Point':
                    intersection_point = intersect_line.coords[0]
                #Check that there's no division by 0
                if np.linalg.norm(velocity_vector) < 0.00001:
                    x = old[0]
                    y = old[1]
                else:
                    velocity_vector = radius * (velocity_vector/np.linalg.norm(velocity_vector))
                    x = intersection_point[0] - velocity_vector[0]
                    y = intersection_point[1] - velocity_vector[1]
            lines.append(line)
    return x, y, direction

def along_wall(vel_vec, wall_vec):
    norm_wall = wall_vec/np.linalg.norm(wall_vec)
    new_vel = norm_wall * np.dot(vel_vec, norm_wall)
    return new_vel

def updatescreen(x,y,direction):
    global SENSOR_LINES

    #reset screen
    win.fill((255,255,255))

    initializeEnvironment()

    #draw robot
    pygame.draw.circle(win, (255, 0, 0), (int(round(x)), int(round(y))), radius,radius)

    #draw sensors
    for i in range(0,NUM_OF_SENSORS):
        line = SENSOR_LINES[i]
        a = line.coords[0]
        b = line.coords[1]
        pygame.draw.line(win,(0,255,0),a,b,2)
        sensortext = myfont.render(str(int(round(SENSOR_READING[i]))), 1, (0,0,0))
        win.blit(sensortext, (((a[0]+b[0])/2+a[0])/2-10,((a[1]+b[1])/2+a[1])/2-5))

    #motorspeeds
    leftspeedtext = myfont.render(str(int(round(velocity_left))),1,(0,0,0))
    rightspeedtext = myfont.render(str(int(round(velocity_right))), 1, (0, 0, 0))

    win.blit(leftspeedtext,(x - 3 + radius * np.cos(direction+np.pi/2), y - 5 + radius * np.sin(direction+np.pi/2)))
    win.blit(rightspeedtext, (x - 3 + radius * np.cos(direction - np.pi / 2), y - 5 + radius * np.sin(direction - np.pi / 2)))

    pygame.draw.line(win, (0, 0, 0), (int(round(x)), int(round(y))),
                     (x + radius * np.cos(direction), y + radius * np.sin(direction)), 2)

    pygame.display.update()

# update sensor position with updated position
def updateSensors(x, y, direction):
    global SENSOR_LINES, SENSOR_READING
    SENSOR_LINES.clear()
    # create sensor lines with updated position
    for s_ind in range(0, NUM_OF_SENSORS):
        if not s_ind:
            # initial sensor line positioned at new position
            A = Point(x,y)
            B = (x + SENSOR_LINE_LIMIT, y)
            SENSOR_LINES.append(affinity.rotate(LineString([A, B]),  (direction*360/(2*np.pi)), origin=A))
        else:
            # remaining sensor lines created with rotating degrees needed.
            SENSOR_LINES.append(affinity.rotate(SENSOR_LINES[0], ((360 / NUM_OF_SENSORS) * s_ind), origin=A))
    
    # find intersections with walls for each sensor
    for i in range(0, NUM_OF_SENSORS):
        min_distance = 200
        for j in range(0, len(WALLS)):
            if WALLS[j].is_valid and SENSOR_LINES[i].is_valid:
                int_pt = SENSOR_LINES[i].intersection(WALLS[j])
            else:
                int_pt = Point()
            # if valid intersection point calculate distance
            if not int_pt.is_empty:
                distance = A.distance(int_pt)
            else:
                distance = 200
            if distance < min_distance:
                min_distance = distance
        # store measured distances for each sensor
        SENSOR_READING[i] = min_distance
    return SENSOR_LINES

#everything that needs to happen in every frame, as well as giving a possibility to stop the program by closing the window
def everyFrame(x,y,direction):
    run = True
    pygame.time.delay(frametime)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pos = [x, y, direction]
    x, y, direction = keypresses(x, y, direction)
    x, y, direction = check_collision(x, y, direction, pos)
    updateSensors(x, y, direction)
    updatescreen(x, y, direction)
    return x, y, direction, run

#main function, everything gets called from here
def main():
    x = 250
    y = 250
    direction = np.pi
    run = True
    createwalls()

    initialiseWindow()
    while run:
        x, y, direction, run = everyFrame(x, y, direction)
    pygame.quit()

if __name__ == "__main__": main()