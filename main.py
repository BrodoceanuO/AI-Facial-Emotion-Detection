import cv2  # pip install opencv-python
import numpy as np
import tensorflow as tf

#{0: 'Neutral', 1: 'Happy', 2: 'fear', 3: 'Angry'}
path = "FinalModel.h5"
new_model = tf.keras.models.load_model(path)


def webcam_emotion(new_model):
    print("opening webcam...")
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    print("webcam opened")
    while True:
        ret, frame = cap.read()
        # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(faceCascade.empty())
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print("Face not detected")
            else:
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey: ey + eh, ex:ex + ew]  ## cropping the face
                    #face_roi = face_roi/255.
                    final_image = cv2.resize(face_roi, (48, 48))
                    final_image = np.expand_dims(final_image, axis=0)  ## need fourth dimension
                    final_image = final_image / 255.0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    Predictions = new_model.predict(final_image)
                    font_scale = 1.5
                    font = cv2.FONT_HERSHEY_PLAIN
                    if (np.argmax(Predictions) == 0):
                        status = "Neutral"
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        # Draw black background rectangle
                        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                        # Addd text
                        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                    elif (np.argmax(Predictions) == 1):
                        status = "Happy"
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        # Draw black background rectangle
                        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                        # Addd text
                        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                    elif (np.argmax(Predictions) == 2):
                        status = "Fear"
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        # Draw black background rectangle
                        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                        # Addd text
                        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

                    elif (np.argmax(Predictions) == 3):
                        status = "Angry"
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        # Draw black background rectangle
                        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                        # Addd text
                        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.imshow('FER mobilenet v2', frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

webcam_emotion(new_model)