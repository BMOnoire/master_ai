import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris
import copy

# conda install -c menpo opencv

def get_script_variables():
    if len(sys.argv) < 2:
        print("You have to add 2 or more files")
        return []

    argv = sys.argv[1:]

    img_path_list = []
    for arg in argv:
        if arg[0] != "-":
            print("Invalid command [", arg, "] you have to put [-] before the file path")
            return []
        path = Path("./" + arg[1:])

        if not path.exists():
            print("The file [", path, "] does not exist")
            return []
        if path.suffix.lower() != ".jpg" and path.suffix.lower() != ".jpeg" and path.suffix.lower() != ".png":
            print("The file [", path.name, "] has a wrong extension")
            return []

        img_path_list.append(str(path))

    return img_path_list

def get_image_list(img_path_list, filter = None):
    if filter == None:
        return [cv2.imread(str(path))for path in img_path_list]
    else:
        return [cv2.cvtColor(cv2.imread(str(path)), filter) for path in img_path_list]

def show_image(img_src, filter = None):
    img = copy.deepcopy(img_src)
    if filter:
        img = cv2.cvtColor(img, filter)
    plt.imshow(img)
    plt.show()

def show_multi_images(img_list_src, col_max_len, filter = None):
    img_list = copy.deepcopy(img_list_src)
    for k, img in enumerate(img_list):
        #r = (k%max_len) + 1
        row_len = int(k / col_max_len) + 1
        if filter:
            img = cv2.cvtColor(img, filter)
        plt.subplot(row_len, col_max_len, k+1), plt.imshow(img)

    plt.show()

def get_harris_corners(img_src, blocksize, threshold, dilate_corners = False):
    img = copy.deepcopy(img_src)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, blocksize, 3, 0.04)

    #  result is dilated for marking the corners, not important
    if dilate_corners:
        corners = cv2.dilate(corners, None)

    # Threshold for an optimal value, it may vary depending on the image.
    matrix_arg_threshold = corners > threshold * corners.max()
    img[matrix_arg_threshold] = [0, 0, 255]
    keypoints = np.argwhere(matrix_arg_threshold)
    keypoints = [cv2.KeyPoint(kp[1], kp[0], 1) for kp in keypoints]
    return img, keypoints


def get_sift(img_src):
    img = copy.deepcopy(img_src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    sift_img = cv2.drawKeypoints(gray, kp, img)
    return sift_img

def main():
    # get the image path and check if it is all correct
    img_path_list = get_script_variables()
    if img_path_list == []:
        return 1

    img_list = get_image_list(img_path_list)
    #show_multi_images(img_list, 2, cv2.COLOR_BGR2RGB) #  cv2.COLOR_BGR2RGB
    [show_image(k, cv2.COLOR_BGR2RGB) for k in img_list]

    #  (1) find harris corners
    harris_img_list, harris_corners_list = [], []
    for img in img_list:
        harris_img, h_corners_list = get_harris_corners(img, 3, 0.01)
        harris_img_list.append(harris_img)
        harris_corners_list.append(h_corners_list)

    #show_multi_images(harris_img_list, 2)
    [show_image(k, cv2.COLOR_BGR2RGB) for k in harris_img_list]

    # TODO fai il test per la grandezza del corner harris


    #  (2) compute SIFT descriptors from corners
    sift_img_list = []
    for img in img_list:
        sift_img = get_sift(img)
        sift_img_list.append(sift_img)
    #show_multi_images(sift_img_list, 2)
    [show_image(k) for k in sift_img_list]

    # TODO capire se sta scrivendo solo stronzate o meno

    #  (3) compute the distances between every descriptor in image 1 with every descriptor in image 2
    #  (3a) Normalized correlation
    #  (3b) Euclidean distance after normalizing each descriptor

    #  (4)Select the best matches based on a threshold or by considering the top few hundred pairs of descriptors.
    #  (4B) make a sensitivity analysis based on these parameters.

    # (5)

    # (6)

    # (7)

    return 0










if __name__ == "__main__":
    main()
