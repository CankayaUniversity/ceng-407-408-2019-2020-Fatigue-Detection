import cv2
import dlib
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time

def stopwatch(seconds):
    global threshold
    global semptom
    start = time.time()
    time.clock()
    elapsed = 0
    semptom = semptom + yawn_counter + closed_eye_sec_counter + headnod_counter + freq_blinking_counter
    threshold = 0.75
    while elapsed < seconds:
        elapsed = time.time() - start
        if(semptom==0):
            threshold = threshold - 0.08
            elapsed = 0
            semptom=1
        #else:
        #   elapsed = 0
        print ("time %02d" % (elapsed))
        print ("threshold %f" % (threshold))
        time.sleep(1)
stopwatch(15)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    eye_aspect_rat = (A + B) / (2.0 * C)

    return eye_aspect_rat

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    eye_aspect_rat = (A + B) / (2.0 * C)

    return eye_aspect_rat

def final_eye_aspect_ratio(shape):

    (rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

    leftEye = shape[leftStart:leftEnd]
    rightEye = shape[rightStart:rightEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_dist(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    dist = abs(top_mean[1] - low_mean[1])
    return dist

def calculateCounters(msg):
    global alarm_status
    global yawn_counter
    global closed_eye_counter
    global headnod_counter
    global freq_blinking_counter
    #global closed_eye_sec
    while alarm_status:
        yawn_counter+=yawn_counter


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="")
args = vars(ap.parse_args())

eye_threshold = 0.3
eye_consecutive_frame = 30
yawn_threshold = 45
yawn_counter=0
closed_eye_sec_counter=0
COUNTER = 0

predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

videoStream = VideoStream(src=args["webcam"]).start() #
time.sleep(1.0)

while True:

    frame = videoStream.read()
    frame = imutils.resize(frame, width=800,height=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detect.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predict(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_eye_aspect_ratio(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_dist(shape)

        leftEyeH = cv2.convexHull(leftEye)
        rightEyeH = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeH], -1, (0, 255, 0), 1) #leftEye
        cv2.drawContours(frame, [rightEyeH], -1, (0, 255, 0), 1) #rightEye

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1) #lip

        if ear < eye_threshold:
            COUNTER += 1

        else:
            COUNTER = 0
            alarm_status = False

        if (distance > yawn_threshold):
           cv2.putText(frame, "Yawn Alert", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
videoStream.stop()