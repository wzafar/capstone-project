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



# Required Functions

# function to calculate EAR
def calc_ear(eye):
    # calculate the horizontal distance
    hor_dist = dist.euclidean(eye[0],eye[1])
    
    # calculate the 2 vertical distances and get their average
    ver_dist1 = dist.euclidean(eye[2],eye[4])
    ver_dist2 = dist.euclidean(eye[1],eye[5])
    ver_dist = (ver_dist1 + ver_dist2)/2.0
    
    # calculate EAR: vertical distance divided by horizontal distance
    ear = ver_dist/hor_dist
    
    return ear

def play_alarm(path):
    playsound(path)



# Required Variables

frame_counter = 0
flag = False

# experiment with these 2 values
ear_thres = 0.5 # threshold for EAR below which the eyes are considered to be closed
consec_frames = 10 # number of consecutive frames EAR must be below the EAR threshold in order to activate alarm




# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the landmarks predictor to detect the landmarks
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')




# capture frames from webcam/video stream
#cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('../data/videos/pranav.mp4')




# Get the indexes of the facial landmarks for left and right eye
(left_first, left_last) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_first, right_last) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]




ear_data = []
isFlagged = []

# Loop over the frames from video stream
while(cap.isOpened()):
    # read the frame
    ret, frame = cap.read()
    if(ret == True):
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect the face
        faces = detector(gray)

        for face in faces:
            # predict the facial landmarks for the face in the frame
            shape = predictor(gray,face)
            # convert facial landmark coordinates to a numpy array
            shape = face_utils.shape_to_np(shape)
            # extract the eye coordinates from the array of facial landmark coordinates
            left_eye = shape[left_first:left_last]
            right_eye = shape[right_first:right_last]
            # compute EAR for each eye
            left_ear = calc_ear(left_eye)
            right_ear = calc_ear(right_eye)
            # calculate the average EAR
            ear = (left_ear + right_ear)/2.0

            # draw the contours highlighting the eye regions
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            if ear < ear_thres:
                frame_counter += 1

                if frame_counter >= consec_frames:
                    if not flag:
                        flag = True
                        # start a new thread to play the alarm simultaneously with the video feed (parallel processing)
                        t = Thread(target=play_alarm , args=('../alarm_trimmed.mp3',))
                        t.deamon = True
                        t.start()

                    # Display a visual warnning
                    cv2.putText(frame, "FALLING ASLEEP!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                frame_counter = 0
                flag = False

            isFlagged.append(flag)
            ear_data.append(ear)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break
            
cap.release()
cv2.destroyAllWindows()




# save the data
data = pd.DataFrame(list(zip(ear_data,isFlagged)), columns = ['EAR','isFlagged'])
data.to_csv('../data/gt/pranav_dd.csv')






