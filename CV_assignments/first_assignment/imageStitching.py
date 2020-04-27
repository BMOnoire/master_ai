import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris


def harris_corners(image):
  i = 2
  j = 11
  k = 4
  gray_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_copy = np.float32(gray_copy)
  corners = cv2.cornerHarris(gray_copy, i, j, k/20)
  # dst = cv2.dilate(dst, None)
  copy[dst > 0.01 * dst.max()] =  [0, 0, 255]
  # cv2.imwrite('{}-{}-{}.png'.format(i, j, k), copy)
  return [cv2.KeyPoint(corner[0], corner[1], 10) for corner in corners]

def main():
    if len(sys.argv) < 3:
        print("You have to add 2 or more files")
        return 1

    argv = sys.argv[1:]

    imgs = []
    for arg in argv:
        if arg[0] != "-":
            print("Invalid command [", arg, "] you have to put [-] before the file path")
            return 1
        path = Path("./" + arg[1:])
        print(path)
        if not path.exists():
            print("The file [", path, "] does not exist")
            return 1
        if path.suffix.lower() != ".jpg" and path.suffix.lower() != ".jpeg" and path.suffix.lower() != ".png":
            print("The file [", path.name, "] has a wrong extension")
            return 1

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        imgs.append(img)


    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        #asd = harris_corners(img)
        # result is dilated for marking the corners, not important
        #dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [255, 0, 0]

        #cv2.imshow('dst', img)
        #if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()

        plt.imshow(img)
        plt.title("Reading picture")
        plt.show()
        print(img.shape)


if __name__ == "__main__":
    main()


