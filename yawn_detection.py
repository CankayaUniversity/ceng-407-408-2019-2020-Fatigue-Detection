import wx
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame,self).__init__(parent,title =title,size=(400 ,300))


        self.panel= MyPanel(self)

class MyPanel(wx.Panel):
    def __init__(self,parent):
        super(MyPanel,self).__init__(parent)

        self.label2=wx.StaticText(self,label="Welcome to Fatigue Detection System",pos=(90,30))
        self.label = wx.StaticText(self, label=" ", pos=(100, 120))
        self.label3 = wx.StaticText(self, label=" ", pos=(150, 100))
        self.Btn = wx.Button(self, label="Face Camera",pos=(100,80))
        self.Btn.Bind(wx.EVT_BUTTON,self.onBtnClick)
        self.Btn2 = wx.Button(self, label="Road Camera", pos=(200, 80))
        self.Btn2.Bind(wx.EVT_BUTTON, self.onBtn2Click)
        #self.labelfatigue=wx.StaticText(self,label="Fatigue Level: ",pos=(100,200))
        #textCtrl = wx.TextCtrl(self, size=(150, 100), pos=(200, 200))

    #FaceCamera
    def onBtnClick (self, event):
      self.label.SetLabelText("Face Camera is activating please wait...")


      def eye_aspect_ratio(eye):
          A = dist.euclidean(eye[1], eye[5])
          B = dist.euclidean(eye[2], eye[4])

          C = dist.euclidean(eye[0], eye[3])

          eye_aspect_rat = (A + B) / (2.0 * C)

          return eye_aspect_rat

      def final_eye_aspect_rat(shape):
          (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
          (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

          leftEye = shape[lStart:lEnd]
          rightEye = shape[rStart:rEnd]

          leftEAR = eye_aspect_ratio(leftEye)
          rightEAR = eye_aspect_ratio(rightEye)

          eye_aspect_rat = (leftEAR + rightEAR) / 2.0
          return (eye_aspect_rat, leftEye, rightEye)

      def lip_dist(shape):
          lipT = shape[50:53]
          lipT = np.concatenate((lipT, shape[61:64]))

          lipL = shape[56:59]
          lipL = np.concatenate((lipL, shape[65:68]))

          top_mean = np.mean(lipT, axis=0)
          low_mean = np.mean(lipL, axis=0)

          dist = abs(top_mean[1] - low_mean[1])
          return dist

      ap = argparse.ArgumentParser()
      ap.add_argument("-w", "--webcam", type=int, default=0,
                      help="index of webcam on system")
      args = vars(ap.parse_args())

      eye_thresh = 0.28
      eye_frames = 30
      yawn_thresh = 20
      COUNTER = 0
      fatigue = 0
      yawncounter=1


      detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
      predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


      vs = VideoStream(src=args["webcam"]).start()
      time.sleep(1.0)

      while True:

          frame = vs.read()
          frame = imutils.resize(frame, width=450)
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

          for (x, y, w, h) in rects:
              rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

              shape = predictor(gray, rect)
              shape = face_utils.shape_to_np(shape)

              eye = final_eye_aspect_rat(shape)
              ear = eye[0]
              leftEye = eye[1]
              rightEye = eye[2]

              distance = lip_dist(shape)

              leftEyeHull = cv2.convexHull(leftEye)
              rightEyeHull = cv2.convexHull(rightEye)
              cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
              cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

              lip = shape[48:60]
              cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

              if ear < eye_thresh:
                  COUNTER += 1

                  if COUNTER >= eye_frames:
                      fatigue=fatigue+3
                      if alarm_status == False:
                          alarm_status = True

                      cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

              else:
                  COUNTER = 0
                  if fatigue>0.03:
                      fatigue=fatigue-0.03

                  alarm_status = False

              if (distance > yawn_thresh):
                  cv2.putText(frame, "Yawn Alert", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                  if alarm_status2 == False:

                      fatigue=fatigue+yawncounter
                      if(yawncounter<0.15+yawncounter):
                          yawncounter=yawncounter+0.15
                      alarm_status2 = True
              else:
                  alarm_status2 = False

              if fatigue>100:
                  fatigue=100
                  cv2.putText(frame, "Fatigue Level:{:.2f} ".format(fatigue), (250, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
              else:
               cv2.putText(frame, "Fatigue Level:{:.2f} ".format(fatigue), (250, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

          cv2.imshow("Frame", frame)
          key = cv2.waitKey(1) & 0xFF

          if key == ord("q"):
              break

      cv2.destroyAllWindows()
      vs.stop()

    #RoadCamera
    def onBtn2Click(self, event):
        self.label.SetLabelText("Road Camera is activating please wait...")

        def region_of_interest(img, vertices):
            mask = np.zeros_like(img)
            match_mask_color = 255
            cv2.fillPoly(mask, vertices, match_mask_color)
            masked_img = cv2.bitwise_and(img, mask)
            return masked_img

        def drow_the_lines(img, lines):
            img = np.copy(img)
            blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

            img = cv2.addWeighted(img, 0.8, blank_image, 0.7, 0.0)
            return img

        def process(image):
            print(image.shape)
            height = image.shape[0]
            width = image.shape[1]
            region_of_interest_vertices = [
                (0, height),
                (width / 2, height / 1.4),
                (width, height)
            ]
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            canny_image = cv2.Canny(gray_image, 100, 120)
            cropped_image = region_of_interest(canny_image,
                                               np.array([region_of_interest_vertices], np.int32), )
            lines = cv2.HoughLinesP(cropped_image,
                                    rho=2,
                                    theta=np.pi / 180,
                                    threshold=50,
                                    lines=np.array([]),
                                    minLineLength=40,
                                    maxLineGap=100)
            image_with_lines = drow_the_lines(image, lines)
            return image_with_lines

        cap = cv2.VideoCapture('test.mp4')

        while cap.isOpened():
            ret, frame = cap.read()
            frame = process(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()


class MyApp(wx.App):
       def OnInit(self):
        self.frame = MyFrame(parent = None, title="Fatigue Detection")
        self.frame.Show()
        return True


app = MyApp()
app.MainLoop()