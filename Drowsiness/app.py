import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame

def shape_to_np(shape, dtype=int):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    return coordinates

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.wav")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EYE_AR_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 48

frame_count = 0
blink_count = 0
alarm_active = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESHOLD:
            blink_count += 1
            if blink_count >= EYE_AR_CONSEC_FRAMES and not alarm_active:
                alarm.play()
                alarm_active = True
        else:
            alarm_active = False
            blink_count = 0

        # Draw a single rectangle around each eye to make them more visible
        left_eye_box = cv2.boundingRect(np.array([left_eye], np.int32))
        right_eye_box = cv2.boundingRect(np.array([right_eye], np.int32))
        cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), (left_eye_box[0] + left_eye_box[2], left_eye_box[1] + left_eye_box[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), (right_eye_box[0] + right_eye_box[2], right_eye_box[1] + right_eye_box[3]), (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
