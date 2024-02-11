import cv2
import numpy as np
import face_recognition as fr
import os

path = 'FacesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:  # load files into our custom array images[] and classNames[]
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])  # [0] it will grab the first element, not the extension (.jpg) which is last
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color format
        encode = fr.face_encodings(img)[0] # encode
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # 0,0 pixels, 0.25 means 1/4 of the size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # in the webcam image we might actually find multiple faces so for that we need to find the lcation of our faces
    # and send this locations to our face_encodings function

    facesCurrFrame = fr.face_locations(imgS)
    encodesCurrFrame = fr.face_encodings(imgS, facesCurrFrame)

    # one by one it will grab one faceLoc from facesCurrFrame list and then it will grab the encodeFace in the encodesCurrFrame
    for encodeFace, faceLoc in zip (encodesCurrFrame, facesCurrFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        print("Matches: ", matches)
        print("FaceDis: ", faceDis)
        matchIndex = np.argmin(faceDis)
        print("Match Index:", matchIndex)  # Debugging: Print match index
        print("Class Names:", classNames)  # Debugging: Print class names
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)


#ytlink: https://www.youtube.com/watch?v=sz25xxF_AVE&t=281s