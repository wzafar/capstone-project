from math import hypot
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
from playsound import playsound
import imutils
import dlib
import cv2
import pandas as pd

cap = cv2.VideoCapture(0)

isFlagged = []
frame_counter = 0
flag = False
consec_frames = 10

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def play_alarm(path):
    playsound(path)

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    
    

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

while True:
    _, frame = cap.read()
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)


        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2



        if gaze_ratio <= 0.5:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            frame_counter += 1

            if frame_counter >= consec_frames:
                if not flag:
                    flag = True
                    # start a new thread to play the alarm simultaneously with the video feed (parallel processing)
                    t = Thread(target=play_alarm , args=('../alarm_trimmed.mp3',))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "LOOK AT ROAD", (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

        elif gaze_ratio >= 2.2:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            frame_counter += 1

            if frame_counter >= consec_frames:
                if not flag:
                    flag = True
                    # start a new thread to play the alarm simultaneously with the video feed (parallel processing)
                    t = Thread(target=play_alarm , args=('../alarm_trimmed.mp3',))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "LOOK AT ROAD", (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        else:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
            frame_counter = 0
            flag = False




    cv2.imshow("Frame", frame)
    

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


