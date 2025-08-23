import cv2 as cv
import os


def detectAndDisplay(frame):
    # Utiliser le fichier de cascade existant
    cascade_path = os.path.join("requiredFiles", "haarcascade_frontalface_alt.xml")
    face_cascade = cv.CascadeClassifier(cascade_path)
    
    # Vérifier que le modèle a bien été chargé
    if face_cascade.empty():
        print("Erreur: Impossible de charger le modèle Haar Cascade")
        return
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Capture - Face detection', frame)

src = cv.imread("ousmane.jpg")
detectAndDisplay(src)
cv.waitKey(0)
cv.destroyAllWindows()
