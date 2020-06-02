import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_ideal_filter(img, D_0=50, filter_type='low_pass'):
    M, N = img.shape
    mid_U = M / 2
    mid_V = N / 2
    U = np.arange(0, M, 1)
    V = np.arange(0, N, 1)
    U, V = np.meshgrid(U, V)
    H = np.sqrt((U - mid_U) ** 2 + (V - mid_V) ** 2)
    H = H <= D_0
    return H


def get_butterworth_filter(img, D_0=50, filter_type='high_pass', n=1):
    M, N = img.shape
    mid_U = M / 2
    mid_V = N / 2
    U = np.arange(0, M, 1)
    V = np.arange(0, N, 1)
    U, V = np.meshgrid(U, V)
    tmp_d = np.sqrt((U - mid_U) ** 2 + (V - mid_V) ** 2)
    H = 1 / (1 + np.power((D_0 / tmp_d), 2 * n))
    return H


if __name__ == "__main__":
    img = cv2.imread("enhancement-restoration/lab2b.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H = get_butterworth_filter(img)

