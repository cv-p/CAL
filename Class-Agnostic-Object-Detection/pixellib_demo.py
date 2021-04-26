import pixellib
from pixellib.instance import instance_segmentation
import cv2
import os

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
IMG_DIRECTORY = "drive-download/"
DESTINATION = "z/"
for file in os.listdir(IMG_DIRECTORY):
    print(file)
    segmask, output = instance_seg.segmentImage(IMG_DIRECTORY + file, show_bboxes=True)
    cv2.imwrite(DESTINATION + "*" + file, output)
    print(output.shape)
