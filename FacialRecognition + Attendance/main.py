import cv2 #BGR as default color format
import numpy as np
import face_recognition as fr # can only understand RGB color format image

imgElon = fr.load_image_file("Images/elon.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElon = cv2.resize(imgElon,(510,800))
imgElonTest = fr.load_image_file("Images/elon_musk.jpg")
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)
imgHawkingTest = fr.load_image_file("Images/stephen_hawking.jpg")
imgHawkingTest = cv2.cvtColor(imgHawkingTest, cv2.COLOR_BGR2RGB)

#[0] kay we are only sending a single image, we will just get the first element of this
faceLoc = fr.face_locations(imgElon)[0]
encodeElon = fr.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2)

faceLocTest = fr.face_locations(imgElonTest)[0]
encodeElonTest = fr.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]),(255,0,255),2)

faceLocTest2 = fr.face_locations(imgHawkingTest)[0]
encodeHawkingTest = fr.face_encodings(imgHawkingTest)[0]
cv2.rectangle(imgHawkingTest,(faceLocTest2[3], faceLocTest2[0]),(faceLocTest2[1], faceLocTest2[2]),(255,0,255),2)

compare_res1 = fr.compare_faces([encodeElon], encodeElonTest)
compare_res2 = fr.compare_faces([encodeElon], encodeHawkingTest)
faceDis = fr.face_distance([encodeElon], encodeElonTest)
faceDis2 = fr.face_distance([encodeElon], encodeHawkingTest)
print(compare_res1, faceDis)
print(compare_res2, faceDis2)

cv2.putText(imgElonTest,f'{compare_res1}, {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.putText(imgHawkingTest,f'{compare_res2}, {round(faceDis2[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Elon Musk", imgElon)
cv2.waitKey(0)
cv2.imshow("StephenHawking Test Subject", imgHawkingTest)
cv2.waitKey(0)
cv2.imshow("ElonMusk Test Subject", imgElonTest)
cv2.waitKey(0)



