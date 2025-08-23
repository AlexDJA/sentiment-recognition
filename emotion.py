import cv2 as cv

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
   
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
 

cap.release()
cv.destroyAllWindows()