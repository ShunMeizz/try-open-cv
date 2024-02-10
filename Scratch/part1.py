import cv2
import numpy as np
'''
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
'''
# shape[0]- height; shape[1] = width
box = np.zeros((800,1000,3), dtype=np.uint8) # 3 represents the color channel(RGB)
cv2.line(box, (0, 10), (1000, 10), (0, 255, 0), 2)
cv2.line(box, (0, 680), (1000, 680), (255, 255, 0), 2)
cv2.putText(box,"CAMERA",(int(box.shape[1] / 2), int(box.shape[0] / 10)),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,150,150),1)
vid = cv2.VideoCapture(0)

while True:
    success, video = vid.read()
    video = cv2.resize(video, (700,500))
    y_offset = int((box.shape[0]-video.shape[0])/3)
    x_offset = int((box.shape[1]-video.shape[1])/2)
    box[y_offset:y_offset+video.shape[0], x_offset:x_offset+video.shape[1]] = video
    cv2.imshow("Video Box", box)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        print("Camera is off")
        break

vid.release()
cv2.destroyAllWindows()




