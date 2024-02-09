import cv2
import numpy as np

img = cv2.imread("resources/michi.png")
cv2.imshow("Original Image: ", img)
cv2.waitKey(0)

imgCropped = cv2.resize(img,(430,250))
cv2.imshow("Image Cropped: ", imgCropped)
cv2.waitKey(0)

video = cv2.VideoCapture("resources/NaniwaWay.mp4")

while True:
    success, image_seq = video.read()
    cv2.imshow("Image Sequences <-> Video:", image_seq)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break




