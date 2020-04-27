import pygame
import math
import numpy as np
#
current_robot = None
robot_skipped_wall = False
wall_skipped = None
delta_t = 0
SHOW_SENSOR_VALUES = True
#
wall_width = 5
#
robots = []
walls = []
#
LEFT_WHEEL = 0
RIGHT_WHEEL = 1
BOTH_WHEELS = 2
#
gameOver = False
DRAW_GAME = True
max_velocity = 3000.0
#
#debug
SHOW_SENSORS = False
TICK_RATE = 60
INTERMEDIATE_TICK_RATE = TICK_RATE
SHOW_INTERSECTION_POINTS = False
#
#visualisation
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (66, 135, 245)
defaultFont = None
sensor_value_offset = 8
#
#
class Robot:
    color = None
    coordinates = None
    orientation = 0
    orientation_line_end_pos = None
    radius = 0
    v_l = 0
    v_r = 0
    acceleration = 0
    diameter = 0
    sensor_number = 0
    sensor_length = None
    sensor_value_positions = None
    collision_points = None
    sensors = None
#
#
class Line:
    start_pos = None
    end_pos = None
#
#
class Sensor:
    line = None
    value = 0
    collided = False
#
#
def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b
#
#
def calculate_angle():
    x1, y1 = current_robot.coordinates
    x2, y2 = current_robot.orientation_line_end_pos
#
    delta_y = y2 - y1
    delta_x = x2 - x1
#
    return math.atan2(delta_y, delta_x) * 180 / math.pi
#
#
def angle_between_vectors(v1, v2):
    dot = np.dot(v1, v2)
    v1_magnitude = math.sqrt(np.dot(v1, v1))
    v2_magnitude = math.sqrt(np.dot(v2, v2))
#
    return math.acos(dot / (v1_magnitude * v2_magnitude))
#
#
def intersect_t(line1, line2):
    x1, y1, x2, y2, x3, y3, x4, y4 = break_down_lines(line1, line2)
#
    a = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    b = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#
    if b == 0:
        b = 0.0001
#
    return a / b
#
#
def intersect_u(line1, line2):
    x1, y1, x2, y2, x3, y3, x4, y4 = break_down_lines(line1, line2)
#
    a = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    b = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#
    if b == 0:
        b = 0.0001
#
    return -(a / b)
#
#
def intersection_point(line1, line2):
    x1, y1, x2, y2, x3, y3, x4, y4 = break_down_lines(line1, line2)
    t = intersect_t(line1, line2)
    u = intersect_u(line1, line2)
#
    if not (0.0 <= t <= 1.0):
        return None
    if not (0.0 <= u <= 1.0):
        return None
#
    Px = x1 + t * (x2 - x1)
    Py = y1 + t * (y2 - y1)
#
    return Px, Py
#
#
def debug_mode():
    global SHOW_SENSORS
    global TICK_RATE
    global SHOW_INTERSECTION_POINTS
#
    SHOW_SENSORS = True
    TICK_RATE = 30
    SHOW_INTERSECTION_POINTS = True
#
#
def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
#
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
#
#
def has_skipped_wall(wall, new_position):
    global current_robot
    if np.array_equal(new_position, current_robot.coordinates):
        return False
    line_traveled = Line()
    line_traveled.start_pos = current_robot.coordinates
    line_traveled.end_pos = new_position
    return intersection_point(line_traveled, wall) is not None
#
#
def calculate_R():
    a = current_robot.v_l + current_robot.v_r
    b = current_robot.v_r - current_robot.v_l
#
    return current_robot.diameter / 2 * (a / b)
#
#
def calculate_w():
    return (current_robot.v_r - current_robot.v_l) / current_robot.diameter
#
#
def calculate_ICC(R, angle):
    x, y = current_robot.coordinates
#
    ICC_x = x - R * math.sin(angle)
    ICC_y = y + R * math.cos(angle)
#
    return np.array([ICC_x, ICC_y])
#
#
def calculate_rotation_matrix(odt):
    return np.array([
        [math.cos(odt), -math.sin(odt), 0],
        [math.sin(odt), math.cos(odt), 0],
        [0, 0, 1],
    ])
#
#
def calculate_forward_kinematics():
    angle = calculate_angle()
    angle = math.radians(angle)
    R = calculate_R()
    ICC = calculate_ICC(R, angle)
    w = calculate_w()
#
    odt = w * delta_t
    rm = calculate_rotation_matrix(odt)
    x, y = current_robot.coordinates
#
    a = np.array([x - ICC[0], y - ICC[1], angle]).reshape(3, 1)
#
    b = np.array([ICC[0], ICC[1], odt]).reshape((3, 1))
#
    return np.dot(rm, a) + b
#
#
def translate_sensors(translation):
    global current_robot
    for i in range(len(current_robot.sensors)):
        sensor = current_robot.sensors[i]
        sensor.line.start_pos = sensor.line.start_pos + translation
        sensor.line.end_pos = sensor.line.end_pos + translation
        current_robot.sensor_value_positions[i] = current_robot.sensor_value_positions[i] + translation
#
#
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
#
#
def is_point_on_line(line1, point):
    a = line1.start_pos
    b = line1.end_pos
    c = point
    return distance(a, c) + distance(c, b) == distance(a, b)
#
#
def project_robot_on_line_normal(robot, line_normal):
    return np.dot(robot, line_normal) / np.linalg.norm(line_normal)
#
#
def has_collided_with_walls(r_coordinates):
#    use only if maze has walls that are smaller than the robot
#    are_walls_inside_robot(new_coordinates)
    global robot_skipped_wall
    global wall_skipped
#
    collided_at = []
    for wall in walls:
 #       if has_skipped_wall(wall, r_coordinates):
 #           robot_skipped_wall = True
 #           wall_skipped = wall
 #           return np.array(collided_at)
#
        Q = r_coordinates
        r = current_robot.radius
#
        P1 = wall.start_pos
        V = wall.end_pos - P1
#
        a = V.dot(V)
        b = 2 * V.dot(P1 - Q)
        c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r ** 2
#
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            continue
#
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
#
        if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
            continue
#
        t = max(0, min(1, - b / (2 * a)))
        collided_at.append(P1 + t * V)
#
    return np.array(collided_at)
#
#
def resolve_collisions(new_coordinates, new_orientation):
    has_collided_with_walls(new_coordinates)
#
    points_of_collision = has_collided_with_walls(new_coordinates)
#
    if len(points_of_collision) == 0:
        return new_coordinates, new_orientation
    s = new_coordinates - current_robot.coordinates
    point_of_collision = points_of_collision[0]
    rx, ry = current_robot.coordinates
    x2, y2 = point_of_collision
    angle_r = math.radians(current_robot.orientation)
#
    dx = x2 - rx
    dy = y2 - ry
#
    angle_r_minus_s = math.atan2(dy, dx)
    angle_m_plus_r = angle_r_minus_s + math.radians(90)
    angle_m = angle_m_plus_r - angle_r
#
    a = math.cos(angle_m) * s
#
    new_coordinates = np.array(
        rotate(current_robot.coordinates, current_robot.coordinates + a, angle_m))
#
    new_points_of_collision = has_collided_with_walls(new_coordinates)
    if len(new_points_of_collision) > 0:
        return current_robot.coordinates, new_orientation
#
    return new_coordinates, new_orientation
#
#
def move_robot(new_coordinates, new_orientation):
    global current_robot
    current_robot.coordinates = new_coordinates
#
    if new_orientation is not None:
        current_robot.orientation = new_orientation
#
#
def calculate_robot_movement():
    new_orientation = None
    if current_robot.v_l != current_robot.v_r:
        translation = calculate_forward_kinematics().reshape((1, 3))[0]
        new_orientation = math.degrees(translation[2])
#
        new_coordinates = np.array([translation[0], translation[1]])
    else:
        total_force = np.array([(current_robot.v_l + current_robot.v_r) * delta_t, 0]) / 2
        new_coordinates = np.array(
            rotate(current_robot.coordinates, current_robot.coordinates + total_force,
                   math.radians(current_robot.orientation)))
#
    return new_coordinates, new_orientation
#
#
def break_down_lines(line1, line2):
    x1, y1 = line1.start_pos
    x2, y2 = line1.end_pos
#
    x3, y3 = line2.start_pos
    x4, y4 = line2.end_pos
#
    return x1, y1, x2, y2, x3, y3, x4, y4
#
#
def update_sensors():
    for sensor in current_robot.sensors:
        sensor.value = current_robot.sensor_length[0]
        for wall in walls:
            line1 = sensor.line
            line2 = wall
            point = intersection_point(line1, line2)
            sensor.collided = False
            if point is None or np.isnan(point).any():
                continue
            sensor.value = np.linalg.norm(point - line1.start_pos)
            if sensor.value <= current_robot.sensor_length[0]:
                sensor.collided = True
                if SHOW_INTERSECTION_POINTS:
                    record_intersection_points(point)
                break
            else:
                sensor.value = current_robot.sensor_length[0]
#
#
def record_intersection_points(point):
    current_robot.collision_points.append([int(point[0]), int(point[1])])
#
#
def increment_wheel_speed(wheel):
    change_wheel_speed(wheel, current_robot.acceleration)
#
#
def decrement_wheel_speed(wheel):
    change_wheel_speed(wheel, -current_robot.acceleration)
#
#
def stop_wheels():
    current_robot.v_l = 0
    current_robot.v_r = 0
#
#
def change_wheel_speed(wheel, value):
    if wheel == LEFT_WHEEL:
        current_robot.v_l += value
    if wheel == RIGHT_WHEEL:
        current_robot.v_r += value
    if wheel == BOTH_WHEELS:
        current_robot.v_l += value
        current_robot.v_r += value
#
    current_robot.v_l = min(current_robot.v_l, max_velocity)
    current_robot.v_r = min(current_robot.v_r, max_velocity)
#
#
def process_input(key):
    if key[pygame.K_w]:
        increment_wheel_speed(LEFT_WHEEL)
    if key[pygame.K_s]:
        decrement_wheel_speed(LEFT_WHEEL)
    if key[pygame.K_o]:
        increment_wheel_speed(RIGHT_WHEEL)
    if key[pygame.K_l]:
        decrement_wheel_speed(RIGHT_WHEEL)
    if key[pygame.K_t]:
        increment_wheel_speed(BOTH_WHEELS)
    if key[pygame.K_g]:
        decrement_wheel_speed(BOTH_WHEELS)
    if key[pygame.K_x]:
        stop_wheels()
#
#
def handle_controls():
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            global gameOver
            gameOver = True
    process_input(pygame.key.get_pressed())
#
#
def draw_robot_body():
    x, y = current_robot.coordinates
#
    x = int(x)
    y = int(y)
#
    pygame.draw.circle(screen, current_robot.color, [x, y], current_robot.radius)
    pygame.draw.circle(screen, BLACK, [x, y], current_robot.radius + 1, 1)
#
#
def draw_robot_orientation():
    x, y = current_robot.coordinates
    line_end_pos = [x + current_robot.radius, y]
    line_end_pos = rotate(current_robot.coordinates, line_end_pos, math.radians(current_robot.orientation))
#
    current_robot.orientation_line_end_pos = line_end_pos
#
    pygame.draw.line(screen, BLACK, current_robot.coordinates, line_end_pos, 2)
#
#
def get_sensor_value_pos(robot, rotation):
    return np.array(rotate(robot.coordinates - sensor_value_offset, robot.coordinates +
                           np.array([robot.radius, -8]),
                           rotation))
#
#
def create_sensors(robot):
    for _ in range(robot.sensor_number):
        sensor = Sensor()
        sensor.line = Line()
        sensor.value = robot.sensor_length[0]
        robot.sensors.append(sensor)
#
#
def place_sensors():
    current_robot.sensor_value_positions.clear()
    sensor_angle = 360 / current_robot.sensor_number
    for x in range(current_robot.sensor_number):
        sensor = current_robot.sensors[x]
        line = sensor.line
#
        sensor_rotation = math.radians(current_robot.orientation + sensor_angle * x)
#
        line_start_pos = current_robot.coordinates + np.array([current_robot.radius, 0])
        line_start_pos = np.array(rotate(current_robot.coordinates, line_start_pos, sensor_rotation))
        line_end_pos = line_start_pos + current_robot.sensor_length
        line_end_pos = np.array(rotate(line_start_pos, line_end_pos, sensor_rotation))
#
        sensor_value_pos = get_sensor_value_pos(current_robot, sensor_rotation)
        current_robot.sensor_value_positions.append(sensor_value_pos)
#
        line.start_pos = line_start_pos
        line.end_pos = line_end_pos
#
        sensor.line = line
#
#
def draw_sensors():
    width = 0
#
    if SHOW_SENSORS:
        width = 5
#
    for sensor in current_robot.sensors:
        color = GREEN
        if sensor.collided:
            color = RED
        pygame.draw.line(screen, color, sensor.line.start_pos, sensor.line.end_pos, width)
#
#
def draw_motor_speed():
    left_motor_speed_pos = current_robot.coordinates + np.array([-1, 0])
    right_motor_speed_pos = current_robot.coordinates + np.array([-1, -20])
#
    left_value_surface = defaultFont.render(str(round(current_robot.v_l, 1)), False, BLACK)
    right_value_surface = defaultFont.render(str(round(current_robot.v_r, 1)), False, BLACK)
#
    screen.blit(left_value_surface, left_motor_speed_pos)
    screen.blit(right_value_surface, right_motor_speed_pos)
#
#
def draw_sensor_values():
    if not SHOW_SENSOR_VALUES:
        return
    for x in range(len(current_robot.sensors)):
        sensor = current_robot.sensors[x]
        pos = current_robot.sensor_value_positions[x]
        sensor_value_surface = defaultFont.render(str(int(sensor.value)), False, BLACK)
        screen.blit(sensor_value_surface, pos)
#
#
def draw_robot():
    draw_sensors()
    draw_robot_body()
    draw_motor_speed()
    draw_sensor_values()
    draw_robot_orientation()
#
#
def create_maze():
    line1 = Line()
    line1.start_pos = np.array([0, 0])
    line1.end_pos = np.array([100, 100])
#
    line2 = Line()
    line2.start_pos = np.array([100, 100])
    line2.end_pos = np.array([200, 100])
#
    line3 = Line()
    line3.start_pos = np.array([200, 100])
    line3.end_pos = np.array([200, 150])
#
    line4 = Line()
    line4.start_pos = np.array([200, 150])
    line4.end_pos = np.array([300, 150])
#
    walls.append(line1)
    walls.append(line2)
    walls.append(line3)
    walls.append(line4)
#
    line5 = Line()
    line5.start_pos = np.array([0, 450])
    line5.end_pos = np.array([100, 350])
#
    line6 = Line()
    line6.start_pos = np.array([100, 350])
    line6.end_pos = np.array([200, 350])
#
    line7 = Line()
    line7.start_pos = np.array([200, 350])
    line7.end_pos = np.array([200, 300])
#
    line8 = Line()
    line8.start_pos = np.array([200, 300])
    line8.end_pos = np.array([300, 300])
#
    walls.append(line5)
    walls.append(line6)
    walls.append(line7)
    walls.append(line8)
#
    line9 = Line()
    line9.start_pos = np.array([10, 10])
    line9.end_pos = np.array([680, 10])
#
    line10 = Line()
    line10.start_pos = np.array([680, 10])
    line10.end_pos = np.array([680, 480])
#
    line11 = Line()
    line11.start_pos = np.array([680, 480])
    line11.end_pos = np.array([10, 480])
#
    line12 = Line()
    line12.start_pos = np.array([10, 480])
    line12.end_pos = np.array([10, 10])
#
    walls.append(line9)
    walls.append(line10)
    walls.append(line11)
    walls.append(line12)
#
#
def draw_maze():
    for wall in walls:
        pygame.draw.line(screen, GREEN, wall.start_pos, wall.end_pos, wall_width)
#
#
def draw_collision_points():
    for point in current_robot.collision_points:
        pygame.draw.circle(screen, RED, point, 6)
    current_robot.collision_points.clear()
#
#
def create_robot():
    robot = Robot()
    robot.color = BLUE
    robot.coordinates = np.array([160, 200])
    robot.orientation = -90  #parallel to the x-axis
    robot.orientation_line_end_pos = [0, 0]
    robot.radius = 25
    robot.v_l = 0.0
    robot.v_r = 0.0
    robot.acceleration = 1.0
    robot.diameter = robot.radius * 2
    robot.sensor_number = 12
    robot.sensor_length = np.array([50, 0])
    robot.sensor_value_positions = []
    robot.collision_points = []
    robot.sensors = []
#
    return robot
#
#
if __name__ == "__main__":
    debug_mode()
    TICK_RATE = 60
    delta_t = 1 / TICK_RATE
#
    pygame.init()
#
    size = (700, 500)
    screen = pygame.display.set_mode(size)
    pygame.font.init()
    defaultFont = pygame.font.SysFont('Comic Sans MS', 12)
    pygame.display.set_caption("Give me a 10 Rico")
#
    for l in range(1):
        c_r = create_robot()
        c_r.coordinates = np.array([200 + (90 * l), 200])
        c_r.sensor_number = 12
#
        create_sensors(c_r)
        robots.append(c_r)
#
    #robots[1].color = BLACK
#    robots[2].color = RED
 #   robots[3].color = WHITE
  #  robots[4].color = GREEN
#
    create_maze()
#
    clock = pygame.time.Clock()
    INTERMEDIATE_TICK_RATE = TICK_RATE
#
    while not gameOver:
        screen.fill(WHITE)
   #     --- Game logic should go here
        for c_r in range(len(robots)):
            current_robot = robots[c_r]
            INTERMEDIATE_TICK_RATE = TICK_RATE
            delta_t = 1 / INTERMEDIATE_TICK_RATE
            robot_skipped_wall = False
            handle_controls()
            coordinates, orientation = calculate_robot_movement()
            coordinates, orientation = resolve_collisions(coordinates, orientation)
            move_robot(coordinates, orientation)
            place_sensors()
            update_sensors()
#
            if not DRAW_GAME:
                continue
#
            draw_robot()
            draw_maze()
            draw_collision_points()
        if not DRAW_GAME:
            continue
        fps = clock.get_fps()
        fps_display = defaultFont.render(str(int(clock.get_fps())), True, BLACK)
        screen.blit(fps_display, (size[0] - 30, size[1] - 30))
#
        #--- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
#
        #--- Limit to X frames per second
        clock.tick(TICK_RATE)
#
#    Once we have exited the main program loop we can stop the game engine:
    pygame.quit()
