import cv2
import numpy as np
import matplotlib.pyplot as plt

LETTER = "e"

def main_b():
    size = 100
    x = np.random.rand(size)
    y = np.power(x, 2)

    #create A matrix 100x100 with y as rows
    A = np.tile(y, (size, 1))
    #A = [y] * 100 other possibility

    plt.imshow(A)
    plt.title("Random picture")
    plt.show()

    plt.imshow(A.astype(int))
    plt.show()

def main_c():
    image_path = ".\\SecondWeekExercises\\exercise1_basics\\lab1a.png"
    img = cv2.imread(image_path)  # reads an image in the BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    plt.imshow(img)
    plt.title("Reading picture")
    plt.show()
    print(img.shape)

def main_d():
    image_path = ".\\SecondWeekExercises\\exercise1_basics\\lab1a.png"
    img = cv2.imread(image_path, 0)  # reads an image in gray
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR -> GRAY

    plt.imshow(img, cmap='gray')
    plt.title("Gray picture")
    plt.show()
    print(img.shape)

def main_e():
    image_path = ".\\SecondWeekExercises\\exercise1_basics\\lab1a.png"
    img = cv2.imread(image_path)  # reads an image in the BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR -> HSV
    print(img.shape)
    plt.subplot(2, 2, 1), plt.imshow(img), plt.title('Original')

    img_h = img[:, :, 0]
    img_s = img[:, :, 1]
    img_v = img[:, :, 2]

    plt.subplot(2, 2, 2), plt.imshow(img_h), plt.title('channel H')
    plt.subplot(2, 2, 3), plt.imshow(img_s), plt.title('channel S')
    plt.subplot(2, 2, 4), plt.imshow(img_v), plt.title('channel V')

    # imgplot = plt.imshow(img,)
    plt.show()
    print(img.shape)

def main_f():
    pass

def main_g():
    pass

###############MAIN#################
if LETTER == "b":
    main_b()
elif LETTER== "c":
    main_c()
elif LETTER == "d":
    main_d()
elif LETTER == "e":
    main_e()
elif LETTER == "f":
    main_f()
elif LETTER == "g":
    main_g()
else:
    print("NO CORRECT LETTER ", LETTER)