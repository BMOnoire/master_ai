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

def show_image(img_src, filter = None, plot = False):
    img = copy.deepcopy(img_src)
    if plot:
        if filter:
            img = cv2.cvtColor(img, filter)
        plt.imshow(img)
        plt.show()
    else:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def show_multi_images(img_list_src):
    img_list = copy.deepcopy(img_list_src)

    # Initiate SIFT detector
    #sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    #kp1, des1 = sift.detectAndCompute(img1, None)
    #kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    #good = []
    #for m, n in matches:
    #    if m.distance < 0.75 * n.distance:
    #        good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2)

    #plt.imshow(img3), plt.show()




def get_harris_corners(img_src, blocksize, threshold, custom_threshold_logic, dilate_corners = False):
    img = copy.deepcopy(img_src)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, blocksize, 3, 0.04)

    '''
    result is dilated for marking the corners, not important
    '''
    if dilate_corners:
        corners = cv2.dilate(corners, None)

    threshold_value = threshold * corners.max()


    matrix_min_threshold = corners < threshold_value
    matrix_max_threshold = corners >= threshold_value

    #asd = copy.deepcopy(img)
    #asd[matrix_max_threshold] = [0, 0, 255]
    #show_image(asd, cv2.COLOR_BGR2RGB)

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

    images = get_image_list(images_name)
    for img in images:
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

def main():
    # get the image path and check if it is all correct
    img_path_list = get_script_variables()
    if img_path_list == []:
        return 1

    img_list = get_image_list(img_path_list)
    img_list = [cv2.resize(img, (0, 0), None, .5, .5) for img in img_list]
    show_multi_images(img_list)
    #show_multi_images(img_list, 2, cv2.COLOR_BGR2RGB) #  cv2.COLOR_BGR2RGB
    # [show_image(k, cv2.COLOR_BGR2RGB) for k in img_list]

    #  (1) find harris corners
    harris_img_list, harris_keypoints_list = [], []
    for img in img_list:
        harris_img, harris_keypoints = get_harris_corners(img, HARRIS_WINDOW_SIZE, 0.01, True)
        harris_img_list.append(harris_img)
        harris_keypoints_list.append(harris_keypoints)

    #show_multi_images(harris_img_list, 2)
    [show_image(k, cv2.COLOR_BGR2RGB) for k in harris_img_list]

    # test_harris()
    # TODO fai il test per la grandezza del corner harris e tira giù le considerazioni

    #  (2) compute SIFT descriptors from corners
    sift_img_list, sift_keypoints_list, sift_descriptors_list = [], [], []
    for img, kp in zip(img_list, harris_keypoints_list):
        sift_img, sift_keypoints, sift_descriptors = get_sift(img, kp)
        sift_img_list.append(sift_img)
        sift_keypoints_list.append(sift_keypoints)
        sift_descriptors_list.append(sift_descriptors)

    #show_multi_images(sift_img_list, 2)
    [show_image(k) for k in sift_img_list]

    print(sift_keypoints_list[0])
    print(sift_descriptors_list[0][0])

    def normal_correlation(dsc1, dsc2):
        #TODO capire perchè len(desc1)
        norm_desc_1 = (dsc1 - np.mean(dsc1)) / (np.std(dsc1))
        norm_desc_2 = (dsc2 - np.mean(dsc2)) / (np.std(dsc2))
        cc_value = np.correlate(norm_desc_1, norm_desc_2) / len(dsc1)
        print(cc_value)
        return cc_value

    matches = []
    for dsc_a in sift_descriptors_list[0]:
        for dsc_b in sift_descriptors_list[1]:
            match_score = normal_correlation(dsc_a, dsc_b)
            if match_score > MATCH_THRESHOLD:
                matches.append([dsc_a, dsc_b])




    #img3 = cv2.drawMatches(img_list[0], sift_keypoints[0], img_list[1], sift_keypoints[1], matches[:10], flags=2, outImg=None)

    ## Convert keypoints to cv2.Keypoint object
    #cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in sift_keypoints_list[0]]
    #cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in sift_keypoints_list[1]]
#
    #out_img = np.array([])
    #good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in matches]
    #out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    out_img = np.array([])
    #good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(N)]
    out_img = cv2.drawMatches(img_list[0], sift_keypoints_list[0], img_list[1], sift_keypoints_list[1], matches1to2=matches, outImg=out_img)
    plt.imshow(out_img), plt.show()

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
