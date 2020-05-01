import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris
# conda install -c menpo opencv


def get_script_variables():
    if len(sys.argv) < 3:
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


def show_image(img, title = ""):
    plt.imshow(img)
    plt.title(title)
    plt.show()
    #print(img.shape)


def show_multi_images(img_list, col_max_len, BGR2RGB = False):

    for k, img in enumerate(img_list):
        #r = (k%max_len) + 1
        row_len = int(k / col_max_len) + 1
        if BGR2RGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(row_len, col_max_len, k+1), plt.imshow(img)

    # imgplot = plt.imshow(img,)
    plt.show()


def get_harris_corners(img, blocksize, ksize, k, threshold_apha, dilate_corners=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, blocksize, ksize, k) # 2, 11, 0

    #  result is dilated for marking the corners, not important
    if dilate_corners:
        corners = cv2.dilate(corners, None)

    # Threshold for an optimal value, it may vary depending on the image.
    matrix_arg_threshold = corners > threshold_apha * corners.max()
    img[matrix_arg_threshold] = [0, 0, 255]
    keypoints = np.argwhere(matrix_arg_threshold)
    keypoints = [cv2.KeyPoint(kp[1], kp[0], 1) for kp in keypoints]
    return img, keypoints


def get_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp = sift.detect(gray, None)
    sift_img = cv2.drawKeypoints(gray, kp, img)

    show_images(sift_img)


def main():
    # get the image path and check if it is all correct
    img_path_list = get_script_variables()
    if img_path_list == []:
        return 1

    img_list = get_image_list(img_path_list)
    show_multi_images(img_list, 3, True)

    # (1) find harris corners
    harris_img_list, harris_corners_list = [], []
    for img in img_list:
        harris_img, h_corners_list = get_harris_corners(img, 2, 3, 0.04, 0.01)
        harris_img_list.append(harris_img)
        harris_corners_list.append(h_corners_list)

    show_multi_images(harris_img_list, 3, True)
    # TODO fai il test per la grandezza del corner harris

    # (2) compute SIFT descriptors from corners


    # (3)

    # (4)

    # (5)

    # (6)

    # (7)

    return 0










if __name__ == "__main__":
    main()
