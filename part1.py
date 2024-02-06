import cv2 #computervision 2
import numpy as np

print("Package Imported")

#Read (Images, Videos, WebCam)

'''
img = cv2.imread("resources/michi.png")
cv2.imshow("Image Output", img) #will show but dali kaayo, need to put waitKey
cv2.waitKey(0)  #1000ms = 1s; 0 = infinite delay
'''
'''
cap = cv2.VideoCapture("resources/NaniwaWay.mp4")
cap = cv2.VideoCapture(0) #0 idnumber of webCam Object
cap.set(3, 640) #3 idnumber of width
cap.set(4, 480) #4 idnumber of height
cap.set(10, 1000) #10 idnumber of brightness

while True:
    success, image = cap.read()
    cv2.imshow("Video Output", image)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
'''

img = cv2.imread("resources/michi.png")
kernel = np.ones((5,5),np.uint8) #np.ones means we want all the values to be 1; 5,5 the size of the matrix;
                            #type of the object: unsigned integer of 8-bit (which means values can range from 0-255)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0) #the kernelsize must be odd number like 3x3,7x7
imgCanny = cv2.Canny(img, 50, 100)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1) #how many iterations we want our kernel to move around, how much thickness do we actually need
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
cv2.imshow("Gray Image: ", imgGray)
cv2.imshow("Blur Image: ", imgBlur)
cv2.imshow("Canny Image: ", imgCanny)
cv2.imshow("Dialation Image: ", imgDialation)
cv2.imshow("Eroded Image: ", imgEroded)
cv2.waitKey(0)



