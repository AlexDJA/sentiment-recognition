import cv2 as cv
import os
import numpy as np
from huggingface_hub import hf_hub_download
from tensorflow import keras


def load_emotion_model(weights_path: str) -> keras.Model:
    base = keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_shape=(48, 48, 3),
    )

    inputs = keras.Input(shape=(48, 48, 3))
    x = base(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(7, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="emotion_vgg16")
    model.load_weights(weights_path)
    return model

def predict_emotion(model, face_region_gray):
    # The model expects (48, 48, 3) float32 in [0, 1]. We have a grayscale crop.
    face = cv.resize(face_region_gray, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.stack([face, face, face], axis=-1)  # (48, 48, 3)
    face = np.expand_dims(face, axis=0)  # (1, 48, 48, 3)

    probs = model.predict(face, verbose=0)[0]
    labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    return labels[int(np.argmax(probs))]

def detectAndDisplay(frame, face_cascade, model):
    
    # Verifying that the model has been loaded
    if face_cascade.empty():
        print("Error loading face cascade")
        return
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face_region = frame_gray[y:y+h, x:x+w]

        # Predict the emotion
        emotion = predict_emotion(model, face_region)
        cv.putText(frame, emotion, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv.imshow('Capture - Face detection', frame)

def main():
    # Load the face cascade
    cascade_path = os.path.join("requiredFiles", "haarcascade_frontalface_alt.xml")
    face_cascade = cv.CascadeClassifier(cascade_path)
    
    # Download the model from the Hugging Face Hub
    model_path = hf_hub_download(repo_id="shivamprasad1001/Emo0.1", filename="Emo0.1.h5")
    model = load_emotion_model(model_path)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        detectAndDisplay(frame, face_cascade, model)

        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()