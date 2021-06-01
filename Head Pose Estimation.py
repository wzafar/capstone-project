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

def drawPolyline(img, shapes, start, end, isClosed=False):
    points = []
    for i in range(start, end + 1):
        point = [shapes.part(i).x, shapes.part(i).y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], isClosed, (255, 80, 0),
                  thickness=1, lineType=cv2.LINE_8)

def draw(img, shapes):
    drawPolyline(img, shapes, 0, 16)
    drawPolyline(img, shapes, 17, 21)
    drawPolyline(img, shapes, 22, 26)
    drawPolyline(img, shapes, 27, 30)
    drawPolyline(img, shapes, 30, 35, True)
    drawPolyline(img, shapes, 36, 41, True)
    drawPolyline(img, shapes, 42, 47, True)
    drawPolyline(img, shapes, 48, 59, True)
    drawPolyline(img, shapes, 60, 67, True)

def play_alarm(path):
    playsound(path)
    
cap = cv2.VideoCapture(0)

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the landmarks predictor to detect the landmarks
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')


isFlagged = []
frame_counter = 0
flag = False
consec_frames = 10


while(cap.isOpened()):
    # read the frame
    ret, frame = cap.read()
    if(ret == True):
        rects = detector(frame)
        for rect in rects:
            print(rect)
            if rect == None:
                cv2.putText(frame, "LOOK AT ROAD", (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            else:
                shape = predictor(frame, rect)
                #print(shape)
                image_points = np.array([[shape.part(30).x, shape.part(30).y], [shape.part(8).x, shape.part(8).y], 
                               [shape.part(36).x, shape.part(36).y], [shape.part(45).x, shape.part(45).y], 
                               [shape.part(48).x, shape.part(48).y], [shape.part(54).x, shape.part(54).y]],dtype='double')
                # 3D model points.
                model_points = np.array([
                                        (0.0, 0.0, 0.0),             # Nose tip
                                        (0.0, -330.0, -65.0),        # Chin
                                        (-225.0, 170.0, -135.0),     # Left eye left corner
                                        (225.0, 170.0, -135.0),      # Right eye right corne
                                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                                        (150.0, -150.0, -125.0)      # Right mouth corner

                                    ])

                draw(frame,shape)


                # Camera internals
                size = frame.shape
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype = "double")

                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), 
                                                                 rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                cv2.line(frame, p1, p2, (255,0,0), 2)

                # calculating euler angles
                rmat, jac = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                #x = np.arctan2(Qx[2][1], Qx[2][2])
                y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
                #z = np.arctan2(Qz[0][0], Qz[1][0])

                if angles[1] < -15:
                    GAZE = "Looking: Left"
                    cv2.putText(frame, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
                elif angles[1] > 15:
                    GAZE = "Looking: Right"
                    cv2.putText(frame, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
                else:
                    GAZE = "Forward"
                    cv2.putText(frame, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)

                cv2.putText(frame, "rotation: {:.2f}".format(y), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if y > 0.6 or y < -0.6:
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
                    frame_counter = 0
                    flag = False

                
                

        #frame_counter += 1
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break
            
cap.release()
cv2.destroyAllWindows()






