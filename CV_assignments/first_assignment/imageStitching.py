import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris

def get_script_variables():
    if len(sys.argv) < 3:
        print("You have to add 2 or more files")
        return []

    argv = sys.argv[1:]

    imgs = []
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

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        imgs.append(img)

    return imgs

def harris_corners(src_img, blocksize, ksize, k, threshold_apha, dilate_corners=False):

    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, blocksize, ksize, k) # 2, 11, 0

    #  result is dilated for marking the corners, not important
    if dilate_corners:
        corners = cv2.dilate(corners, None)

    # Threshold for an optimal value, it may vary depending on the image.
    src_img[corners > threshold_apha * corners.max()] = [255, 0, 0]
    #return [cv2.KeyPoint(corner[0], corner[1], 10) for corner in corners]
    return src_img

def show_images(src_img):
    plt.imshow(src_img)
    plt.title("Image Stitching")
    plt.show()
    print(src_img.shape)

def main():
    imgs = get_script_variables()
    for img in imgs:
        harris_img = harris_corners(img, 2, 3, 0.04, 0.01)
        #TODO fai il test per la grandezza del corner harris
        show_images(harris_img)


if __name__ == "__main__":
    main()
