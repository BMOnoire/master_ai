import numpy as np
import math
from copy import deepcopy
from shapely.geometry import LineString, Point
import shapely
import data as dt

X, Y, TH = 0, 1, 2
V, O = 0, 1


class Localization:
    def __init__(self, init_position):
        self.position = init_position
        self.real_path = [init_position]
        self.last_mu = init_position
        self.mu_path = [[init_position, False]]
        self.last_sigma = np.diagflat([0.01, 0.02, 0.03])
        #self.sigma_path = [self.last_sigma]

        self.matrix_A = np.identity(3)
        self.matrix_C = np.identity(3)
        self.cov_vector_R = [0.035, 0.045, 0.055]
        self.matrix_R = np.diagflat(self.cov_vector_R)  # diagonal array init
        self.cov_vector_Q = [0.065, 0.045, 0.075]
        self.matrix_Q = np.diagflat(self.cov_vector_Q)
        self.z = init_position

    def kalman_filter_prediction(self, position, motion, sensor_estimate, triangulated, dT):
        # PREDICTION

        # predicted MU
        error_epsilon =[0, 0, 0]

        predicted_mu = deepcopy(self.last_mu)
        predicted_mu[X] += math.cos(position[TH]) * dT * motion[V] + np.sqrt(self.cov_vector_R[X])
        predicted_mu[Y] += math.sin(position[TH]) * dT * motion[V] + np.sqrt(self.cov_vector_R[X])
        predicted_mu[TH] += dT * motion[O] + np.sqrt(self.cov_vector_R[X])

        predicted_mu = np.vstack(predicted_mu)

        # predicted SIGMA
        predicted_sigma = self.last_sigma + self.matrix_R
        # print(sensor_estimate)
        if not triangulated:
            return np.hstack(predicted_mu).tolist(), predicted_sigma  # np.hstack to put a horizontal list

        # CORRECTION
        var1 = np.linalg.inv(predicted_sigma + self.matrix_Q)
        matrix_K = predicted_sigma * var1

        # corrected MU
        z = np.vstack(sensor_estimate)
        corrected_mu = predicted_mu + np.dot(matrix_K, (z - predicted_mu))

        # corrected SIGMA
        corrected_sigma = np.dot((np.identity(3) - matrix_K), predicted_sigma)

        return np.hstack(corrected_mu).tolist(), corrected_sigma

    def exact_pose(self, landmarks):
        sensor1, sensor2, sensor3 = landmarks[0], landmarks[1], landmarks[2]  # [0 = [x,y], 1 = distance]]
        A = 2 * sensor2[0][0] - 2 * sensor1[0][0]
        B = 2 * sensor2[0][1] - 2 * sensor1[0][1]
        C = sensor1[1] ** 2 - sensor2[1] ** 2 - sensor1[0][0] ** 2 + sensor2[0][0] ** 2 - sensor1[0][1] ** 2 + \
            sensor2[0][1] ** 2
        D = 2 * sensor3[0][0] - 2 * sensor2[0][0]
        E = 2 * sensor3[0][1] - 2 * sensor2[0][1]
        F = sensor2[1] ** 2 - sensor3[1] ** 2 - sensor2[0][0] ** 2 + sensor3[0][0] ** 2 - sensor2[0][1] ** 2 + \
            sensor3[0][1] ** 2
        x = (C * E - F * B) / (E * A - B * D)
        y = (C * D - A * F) / (B * D - A * E)
        return x, y

    def triangulation(self, landmarks, position):
        # print(landmarks)
        # print(position)
        features = []
        radius = 0.5 * dt.ROBOT_RADIUS
        robot_center = Point(position[X], position[Y]).buffer(1)
        robot_shape = shapely.affinity.scale(robot_center, radius, radius)
        x, y = self.get_robot_direction_point(position)  # exact pose
        alfa = np.degrees(np.arctan2((y - position[1]), (x - position[0])))
        current_theta = 0

        def exact_degree(alfa):
            return (alfa + 360) % 360


        for line in landmarks:
            intersection_point = robot_shape.intersection(line[0])
            # pygame.draw.circle(self.screen, (255, 100, 145), (self.round_point(intersection_point.coords[1])), 10, 2)
            beta = np.degrees(np.arctan2((intersection_point.coords[1][1] - position[1]),
                                         (intersection_point.coords[1][0] - position[
                                             0])))  # TODO remove this part and implement only via beacons
            # with no intersection point
            fi = beta - alfa
            fi = exact_degree(fi)
            # print(theta)
            features.append([line[2][0], line[1]])  # [[x,y],distance]
            current_theta += fi
        # x,y = self.exact_pose(features)
        print("calculated theta", current_theta/3, "real ", exact_degree(alfa))
        avg = current_theta/3
        z = [x, y, np.radians(avg)]
        # print(features)
        # self.z =

        return z

    def get_robot_direction_point(self, position, radius=1):
        x = position[X] + (radius * math.cos(position[2]) * 1)
        y = position[Y] + (radius * math.sin(position[2]) * 1)
        return (x, y)

    def update_localization(self, position, motion, z, triangulated, dT):
        self.real_path.append(position)

        current_mu, current_sigma = self.kalman_filter_prediction(position, motion, z, triangulated, dT)
        # TODO inizio fai qualcosa

        self.mu_path.append([current_mu, triangulated])  # add path of mu and if its triangulated in that moment
        #self.sigma_path.append(current_sigma)
        # TODO fine fai qualcosa
        self.last_mu = current_mu
        self.last_sigma = current_sigma
