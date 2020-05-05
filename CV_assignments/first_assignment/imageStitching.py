import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris
import copy
from skimage.morphology import octagon
from skimage.feature import (peak_local_max, corner_fast, corner_peaks, corner_orientations)
import numpy as np
import cv2
from matplotlib import pyplot as plt

# conda install -c menpo opencv

HARRIS_WINDOW_SIZE = 3
MATCH_THRESHOLD = 0.5


def get_script_variables():
    if len(sys.argv) != 3:
        print("You have to add 2 images")
        return None, None

    arg1, arg2 = sys.argv[1], sys.argv[2]

    def check_arg(arg):
        if arg[0] != "-":
            print("Invalid command [", arg, "] you have to put [-] before the file path")
            return None
        path = Path("./" + arg[1:])
        if not path.exists():
            print("The file [", path, "] does not exist")
            return None
        if path.suffix.lower() != ".jpg" and path.suffix.lower() != ".jpeg" and path.suffix.lower() != ".png":
            print("The file [", path.name, "] has a wrong extension")
            return None
        return str(path)

    img1 = check_arg(arg1)
    img2 = check_arg(arg2)

    if img1 == None or img2== None:
        return None, None

    return img1, img2


def get_image(img_path, filter = None):
    if filter == None:
        return cv2.imread(img_path)
    else:
        return cv2.cvtColor(cv2.imread(img_path), filter)


def show_image(img_src, filter = None, using_plot = False):
    img = copy.deepcopy(img_src)
    if using_plot:
        if filter:
            img = cv2.cvtColor(img, filter)
        plt.imshow(img)
        plt.show()
    else:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_harris_corners(img_src, blocksize, threshold, custom_threshold_logic, dilate_corners = False):
    img = copy.deepcopy(img_src)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, blocksize, 3, 0.04)


    # result is dilated for marking the corners, not important
    if dilate_corners:
        corners = cv2.dilate(corners, None)

    threshold_value = threshold * corners.max()

    matrix_min_threshold = corners < threshold_value
    matrix_max_threshold = corners >= threshold_value

    if custom_threshold_logic:
        #  put corner matrix values to 0 if less than a threshold
        corners[matrix_min_threshold] = 0
        coordinates_local_max = peak_local_max(corners, min_distance = blocksize) # 1

        # update max threshold matrix
        matrix_max_threshold = np.empty(matrix_max_threshold.shape, dtype=bool)
        for cord in coordinates_local_max:
            matrix_max_threshold[cord[0]][cord[1]] = True

    # Threshold for an optimal value, it may vary depending on the image.
    img[matrix_max_threshold] = [0, 0, 255]
    keypoints = np.argwhere(matrix_max_threshold)
    keypoints = [cv2.KeyPoint(kp[1], kp[0], 1) for kp in keypoints]
    return img, keypoints


def test_harris():
    images_name = [
        "images/test1.jpeg",
        "images/test2.jpeg",
        "images/img1.png",
        "images/img2.png"
    ]
    range = [1, 2, 3, 5, 10, 15, 20, 30, 50]

    for img in images_name:
        img = get_image(img)
        for size in range:
            harris_img, h_corners_list = get_harris_corners(img, size, 0.001, True)
            show_image(harris_img, cv2.COLOR_BGR2RGB)
            print("Image [ ", img, " ]: ", len(h_corners_list), "corners")


def get_sift(img_src, harris_keypoints):
    img = copy.deepcopy(img_src)
    kp = harris_keypoints
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #sift = cv2.xfeatures2d.SIFT_create()
    #kp = sift.detect(gray, None)
    #sift_img = cv2.drawKeypoints(gray, kp, img)
    #return sift_img

    sift = cv2.xfeatures2d.SIFT_create()
    sift_keypoints, sift_descriptors = sift.compute(gray, kp)
    sift_img = cv2.drawKeypoints(gray, sift_keypoints, img)
    #print("norm", len(kps), des[0])
    return sift_img, sift_keypoints, sift_descriptors


def get_matches(desc_list_1, desc_list_2, threshold):

    def normal_correlation(desc1, desc2):
        norm_desc_1 = (desc1 - np.mean(desc1)) / np.std(desc1)
        norm_desc_2 = (desc2 - np.mean(desc2)) / np.std(desc2)
        cc_value = np.correlate(norm_desc_1, norm_desc_2) / (len(desc1) - 1)
        if cc_value[0] > 1 or cc_value[0] < -1:
            asd = 1
        return cc_value[0]

    matches = []
    id = 1
    for index_1, desc_1 in enumerate(desc_list_1):
        best_score = -2
        distance = None
        for index_2, desc_2 in enumerate(desc_list_2):

            next_score = normal_correlation(desc_1, desc_2)


            if next_score >= best_score:
                best_score = next_score
                distance = np.linalg.norm(desc_2 - desc_1)

        if best_score >= threshold:


            match = {
                "id": id,
                "index_1": index_1,
                "index_2": index_2,
                "score": best_score,
                "distance": distance
            }
            matches.append(match)
            id = id + 1

    return matches


def test_match_threshold():
    # TODO
    pass



def main():
    # get the image path and check if it is all correct
    img_path_1, img_path_2 = get_script_variables()

    if img_path_1 == None or img_path_2 == None:
        return 1

    img_1, img_2 = get_image(img_path_1), get_image(img_path_2)

    img_1, img_2 = cv2.resize(img_1, (0, 0), None, .5, .5), cv2.resize(img_2, (0, 0), None, .5, .5)

    show_image(img_1, cv2.COLOR_BGR2RGB)
    show_image(img_2, cv2.COLOR_BGR2RGB)

    #  (1A) find harris corners
    harris_img_list, harris_keypoints_list = [], []

    harris_img_1, harris_keypoints_1 = get_harris_corners(img_1, HARRIS_WINDOW_SIZE, 0.01, True)
    harris_img_2, harris_keypoints_2 = get_harris_corners(img_2, HARRIS_WINDOW_SIZE, 0.01, True)

    #print(len(harris_keypoints_1))
    #print(len(harris_keypoints_2))

    show_image(harris_img_1, cv2.COLOR_BGR2RGB)
    show_image(harris_img_2, cv2.COLOR_BGR2RGB)

    #  (1B) test thresholds for harris corners
    # TODO fai il test per la grandezza del corner harris e tira giù le considerazioni
    # test_harris()


    #  (2) compute SIFT descriptors from corners
    sift_img_1, sift_keypoints_1, sift_descriptors_1 = get_sift(img_1, harris_keypoints_1)
    sift_img_2, sift_keypoints_2, sift_descriptors_2 = get_sift(img_2, harris_keypoints_2)

    show_image(sift_img_1, cv2.COLOR_BGR2RGB)
    show_image(sift_img_2, cv2.COLOR_BGR2RGB)


    #  (3) compute the distances between every descriptor in image 1 with every descriptor in image 2  (mormalized correlation and Euclidean distance)

    matches = get_matches(sift_descriptors_1, sift_descriptors_2, MATCH_THRESHOLD)


    #  (4A) select the best matches based on a threshold or by considering the top few hundred pairs of descriptors.

    #  (4B) make a sensitivity analysis based on these parameters.
    test_match_threshold()

    # (5) simple implementation of RANSAC



    # (6)
    # (7)
    return 0


if __name__ == "__main__":
    main()
