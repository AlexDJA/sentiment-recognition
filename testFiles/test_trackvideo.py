import cv2 as cv
import os
import numpy as np

def detectAndDisplay(frame, face_cascade):
    
    # Vérifier que le modèle a bien été chargé
    if face_cascade.empty():
        print("Error loading face cascade")
        return
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face_region = frame_gray[y:y+h,x:x+w]
        face_region = cv.resize(face_region, (48, 48))
        face_region = cv.normalize(face_region, None, 0, 1)

    cv.imshow('Capture - Face detection', frame)

cascade_path = os.path.join("requiredFiles", "haarcascade_frontalface_alt.xml")
face_cascade = cv.CascadeClassifier(cascade_path)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
   
    detectAndDisplay(frame, face_cascade)

    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()







