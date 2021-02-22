import cv2
import numpy as np
import os


def poly_area(X, Y, n):
    j, area = n - 1, 0
    for i in range(0, n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i
    return int(abs(area / 2.0))


AREA_THRESHOLD_MIN = 0.05
AREA_THRESHOLD_MAX = 1.00
IMG_DIRECTORY = "drive-download/"
DESTINATION = "z/"
img_to_boxes = dict()
box_sizes = list()
error_log = list()
for file in os.listdir(IMG_DIRECTORY):
    try:
        img = cv2.pyrDown(cv2.imread(IMG_DIRECTORY + file, cv2.IMREAD_UNCHANGED))
        height, width = img.shape[0], img.shape[1]
        img_area = height * width

        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = list()
        for c in contours:
            vertices = np.int0(cv2.boxPoints(cv2.minAreaRect(c)))
            transposed = np.transpose(vertices)
            area = poly_area(transposed[0], transposed[1], len(vertices))
            if AREA_THRESHOLD_MIN < (area / img_area) < AREA_THRESHOLD_MAX:
                bounding_boxes.append(vertices)
                box_sizes.append(area / img_area)
            # cv2.drawContours(img, [vertices], 0, (0, 0, 255))
        img_to_boxes[IMG_DIRECTORY + file] = bounding_boxes.copy()
        ct = 0
        for box in bounding_boxes:
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array([box])
            cv2.fillPoly(mask, points, (255))
            res = cv2.bitwise_and(img, img, mask=mask)
            rect = cv2.boundingRect(points)
            cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            cv2.imwrite(DESTINATION + "*" + str(ct) + file, cropped)
            ct += 1
    except Exception as e:
        error_log.append(e)

t_images = len(img_to_boxes)
wo_box = sum(len(img_to_boxes[x]) == 0 for x in img_to_boxes)
print("Total Images:", t_images)
print("Images w/o Bounding Box:", wo_box)
print("Percentage:", wo_box / t_images)
print("Avg size of bounding box (% of img size):", sum(box_sizes) / len(box_sizes))






