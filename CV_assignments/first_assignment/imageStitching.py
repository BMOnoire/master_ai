import sys
from pathlib import Path
import copy
from skimage.feature import peak_local_max
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

# conda install -c menpo opencv

RESIZE = 0
HARRIS_WINDOW_SIZE = 3
MATCH_THRESHOLD = 0.5
SHOW_ALL = False

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
        coordinates_local_max = peak_local_max(corners, min_distance=blocksize)

        # update max threshold matrix
        matrix_max_threshold = np.zeros(matrix_max_threshold.shape, dtype=bool)
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

    sift = cv2.xfeatures2d.SIFT_create()
    sift_keypoints, sift_descriptors = sift.compute(gray, kp)
    sift_img = cv2.drawKeypoints(gray, sift_keypoints, img)
    #print("norm", len(kps), des[0])
    return sift_img, sift_keypoints, sift_descriptors


################################################################################
################################################################################
################################################################################
def get_matches_LOFFIA(keypoints_1, keypoints_2, descriptors_1, descriptors_2, ratio, reprojThresh):

    #TODO CHANGE THIS
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(descriptors_1, descriptors_2, 2)


    matches = []
    # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) < 4:
        return None, None, None

    # construct the two sets of points
    ptsA = np.float32([keypoints_1[i] for (_, i) in matches])
    ptsB = np.float32([keypoints_2[i] for (i, _) in matches])

    # compute the homography between the two sets of points
    matrix_H, status = cv2.findHomography( ptsB,ptsA, cv2.RANSAC, reprojThresh)

    # return the matches along with the homograpy matrix and status of each matched point
    return matches, matrix_H, status



def draw_match_lines_LOFFIA(img_1, img_2, keypoints_1, keypoints_2, matches, status):
    # initialize the output visualization image
    h_1, w_1 = img_1.shape[0], img_1.shape[1]
    h_2, w_2 = img_2.shape[0], img_2.shape[1]

    vis = np.zeros((max(h_1, h_2), w_1 + w_2, 3), dtype="uint8")

    vis[0:h_1, 0:w_1] = img_1
    vis[0:h_2, w_2:]  = img_2

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # draw the match
            ptA = (int(keypoints_1[queryIdx][0]), int(keypoints_1[queryIdx][1]))
            ptB = (int(keypoints_2[trainIdx][0]) + w_1, int(keypoints_2[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis
################################################################################
################################################################################
################################################################################


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
    if RESIZE != 0:
        img_1, img_2 = cv2.resize(img_1, (0, 0), None, RESIZE, RESIZE), cv2.resize(img_2, (0, 0), None, RESIZE, RESIZE)

    harris_img_1, harris_keypoints_1 = get_harris_corners(img_1, HARRIS_WINDOW_SIZE, 0.01, True)
    harris_img_2, harris_keypoints_2 = get_harris_corners(img_2, HARRIS_WINDOW_SIZE, 0.01, True)

    if SHOW_ALL:
        #TODO check sometimes the images are wrong but all works fine
        show_image(harris_img_1, cv2.COLOR_BGR2RGB)
        show_image(harris_img_2, cv2.COLOR_BGR2RGB)

    sift_img_1, sift_keypoints_1, sift_descriptors_1 = get_sift(img_1, harris_keypoints_1)
    sift_img_2, sift_keypoints_2, sift_descriptors_2 = get_sift(img_2, harris_keypoints_2)

    if SHOW_ALL:
        show_image(sift_img_1)
        show_image(sift_img_2, cv2.COLOR_BGR2RGB)

    # TODO rivedere da qua
    #matches = get_matches(sift_descriptors_1, sift_descriptors_2, MATCH_THRESHOLD)


    keypoint_1 = np.float32([kp.pt for kp in sift_keypoints_1])
    keypoint_2 = np.float32([kp.pt for kp in sift_keypoints_2])
    ratio = 0.75
    reprojThresh = 4.0

    matches, transformation_matrix, status = get_matches_LOFFIA(keypoint_1, keypoint_2, sift_descriptors_1, sift_descriptors_2, ratio, reprojThresh)
    if status is None:
        return 1

    vis = draw_match_lines_LOFFIA(img_1, img_2, keypoint_1, keypoint_2, matches, status)

    height_1, width_1 = img_1.shape[0], img_1.shape[1]
    height_2, width_2 = img_2.shape[0], img_2.shape[1]
    cv2.imshow("IMAGE 1", img_1)
    cv2.imshow("IMAGE 2", img_2)

    new_size = (width_1 + width_2, height_1)
    result = cv2.warpPerspective(img_2, transformation_matrix, new_size)

    cv2.imshow("STITCHING RESULT", result)
    result[0:height_1, 0:width_1] = img_1
    cv2.imshow("STITCHING RESULT", result)

    # check to see if the keypoint matches should be visualized

    #cv2.imshow("SHOW MATCHES", vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
