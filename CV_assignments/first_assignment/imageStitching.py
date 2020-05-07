import sys
from pathlib import Path
import copy
from skimage.feature import peak_local_max
import numpy as np
import cv2
from matplotlib import pyplot as plt


RESIZE = 0
HARRIS_WINDOW_SIZE = 3
MATCH_THRESHOLD = 0.5
SHOW_CORNERS = False
SHOW_ALL = True
TEST = False


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


def get_sift(img_src, harris_keypoints):
    img = copy.deepcopy(img_src)
    kp = harris_keypoints
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    sift_keypoints, sift_descriptors = sift.compute(gray, kp)
    sift_img = cv2.drawKeypoints(gray, sift_keypoints, img)

    return sift_img, sift_keypoints, sift_descriptors


def get_matches(desc_list_1, desc_list_2, threshold):
    def ncc(d1, d2):
        norm_d1 = (d1 - np.mean(d1)) / np.std(d1)
        norm_d2 = (d2 - np.mean(d2)) / np.std(d2)
        return np.correlate(norm_d1, norm_d2) / (len(d1) - 1)

    # CORRELATION and EUCLIDEAN
    matches = []
    for idx_1, desc_1 in enumerate(desc_list_1):
        for idx_2, desc_2 in enumerate(desc_list_2):

            normal_cross_correlation = ncc(desc_1, desc_2)
            if normal_cross_correlation > 0.5:

                euclidean_distance = np.linalg.norm(desc_1 - desc_2)
                if euclidean_distance < threshold:
                    matches.append((idx_1, idx_2))

    #TODO test primi 100
    if TEST:
        matches = sorted(matches, key=lambda x: x[2], reverse=True)
        matches = matches[0:99]
        matches_kp = []
        for m in matches:
            matches.append((m[0], m[1]))

    return matches


def ransac(keypoints_1, keypoints_2, matches):

    if len(matches) < 4:
        return None, None

    # construct the two sets of points
    pts_1 = np.float32([keypoints_1[i] for (i, _) in matches])
    pts_2 = np.float32([keypoints_2[i] for (_, i) in matches])

    # compute the homography between the two sets of points
    matrix_H, status = cv2.findHomography(pts_2, pts_1, cv2.RANSAC, 4.0)

    # return the matches along with the homograpy matrix and status of each matched point
    return matrix_H, status


def draw_match_lines(img_1, img_2, keypoints_1, keypoints_2, matches, status):
    # initialize the output visualization image
    h_1, w_1 = img_1.shape[0], img_1.shape[1]
    h_2, w_2 = img_2.shape[0], img_2.shape[1]

    img_match_lines = np.zeros((max(h_1, h_2), w_1 + w_2, 3), dtype="uint8")

    img_match_lines[0:h_1, 0:w_1] = img_1
    img_match_lines[0:h_2, w_2:]  = img_2

    # loop over the matches
    for m, s in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            trainIdx, queryIdx = m[0], m[1]
            pt_1 = (int(keypoints_1[trainIdx][0]), int(keypoints_1[trainIdx][1]))
            pt_2 = (int(keypoints_2[queryIdx][0]) + w_2, int(keypoints_2[queryIdx][1]))
            cv2.line(img_match_lines, pt_1, pt_2, (0, 255, 0), 1)

    # return the visualization
    return img_match_lines


def image_stitcher(img_path_1, img_path_2):

    # GET IMAGES
    img_1, img_2 = get_image(img_path_1), get_image(img_path_2)
    if RESIZE != 0:
        img_1, img_2 = cv2.resize(img_1, (0, 0), None, RESIZE, RESIZE), cv2.resize(img_2, (0, 0), None, RESIZE, RESIZE)


    # HARRIS KEYPOINTS
    harris_img_1, harris_keypoints_1 = get_harris_corners(img_1, HARRIS_WINDOW_SIZE, 0.01, True)
    harris_img_2, harris_keypoints_2 = get_harris_corners(img_2, HARRIS_WINDOW_SIZE, 0.01, True)


    # SIFT DESCRIPTORS
    sift_img_1, sift_keypoints_1, sift_descriptors_1 = get_sift(img_1, harris_keypoints_1)
    sift_img_2, sift_keypoints_2, sift_descriptors_2 = get_sift(img_2, harris_keypoints_2)


    # MATCHING
    # get again the couple of pixel
    keypoint_1 = np.float32([kp.pt for kp in sift_keypoints_1])
    keypoint_2 = np.float32([kp.pt for kp in sift_keypoints_2])

    #matches = get_matches_old(sift_descriptors_1, sift_descriptors_2, MATCH_THRESHOLD)
    matches = get_matches(sift_descriptors_1, sift_descriptors_2, MATCH_THRESHOLD)
    if matches is None:
        return 1


    # RANSAC
    transformation_matrix, status = ransac(keypoint_1, keypoint_2, matches)
    if status is None:
        return 1


    # TRANSFORMATION AND WARPING
    img_matching = draw_match_lines(img_1, img_2, keypoint_1, keypoint_2, matches, status)

    height_1, width_1 = img_1.shape[0], img_1.shape[1]
    height_2, width_2 = img_2.shape[0], img_2.shape[1]

    new_size = (width_1 + width_2, height_1)
    img_transformed = cv2.warpPerspective(img_2, transformation_matrix, new_size)
    panorama = copy.deepcopy(img_transformed)
    panorama[0:height_1, 0:width_1] = img_1


    # SHOW IMAGES
    if SHOW_CORNERS:
        cv2.imshow("IMAGE 1", harris_img_1)
        cv2.imshow("IMAGE 2", harris_img_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("IMAGE 1", sift_img_1)
        cv2.imshow("IMAGE 2", sift_img_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if SHOW_ALL:
        cv2.imshow("IMAGE 1", img_1)
        cv2.imshow("IMAGE 2", img_2)
        cv2.imshow("SHOW MATCHES", img_matching)
        cv2.imshow("TRANSFORMATION", img_transformed)

    cv2.imshow("STITCHING RESULT", panorama)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def main():
    # get the image path and check if it is all correct
    img_path_1, img_path_2 = get_script_variables()
    if img_path_1 == None or img_path_2 == None:
        return 1
    return image_stitcher(img_path_1, img_path_2)


if __name__ == "__main__":
    main()
