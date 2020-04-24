import wx
import cv2
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')

class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame,self).__init__(parent,title =title,size=(800 ,600))


        self.panel= MyPanel(self)

class MyPanel(wx.Panel):
    def __init__(self,parent):
        super(MyPanel,self).__init__(parent)

        self.label=wx.StaticText(self,label="No camare active",pos=(150,400))
        self.Btn = wx.Button(self, label="Camera 1",pos=(100,50))
        self.Btn.Bind(wx.EVT_BUTTON,self.onBtnClick)
        self.Btn2 = wx.Button(self, label="Camera 2", pos=(200, 50))
        self.Btn2.Bind(wx.EVT_BUTTON, self.onBtn2Click)


        self.cb1 = wx.CheckBox(self,label="Par 1",pos=(600,400))
        self.cb2 = wx.CheckBox(self, label="Par 2",pos=(650,400))
        self.cb3 = wx.CheckBox(self, label="Par 3",pos=(700,400))
        self.label2=wx.StaticText(self,label="",pos=(600,450))
        self.Bind(wx.EVT_CHECKBOX,self.onChecked)

        Algorithms=["Current Algortihm","Algorithm 1","Algorithm 2","Algorithm 3"]
        self.combobox=wx.ComboBox(self,choices=Algorithms,pos=(450,400))
        self.label3 = wx.StaticText(self, label="", pos=(400, 450))
        self.Bind(wx.EVT_COMBOBOX,self.onCombo)

        textCtrl = wx.TextCtrl(self,size=(250,200),pos=(500,100),style=wx.TE_READONLY)
        self.label4 = wx.StaticText(self, label="Fatigue Level ", pos=(500, 50))

        self.BtnApply = wx.Button(self, label="APPLY", pos=(550,500))
    def onCombo(self,event):
        comboValue=self.combobox.GetValue()
        self.label3.SetLabel("You have choosen"+comboValue)

    def onChecked(self,e):
        cb=e.GetEventObject()
        self.label2.SetLabelText("You have selected "+ cb.GetLabel() +" parameter")


    def onBtnClick (self, event):
      cap = cv2.VideoCapture(0)
      while True:
          ret, frame = cap.read()
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          faces = face_cascade.detectMultiScale(gray, 1.3, 5)
          for (x, y, w, h) in faces:
              cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
              roi_gray = gray[y:y + h, x:x + w]
              roi_color = frame[y:y + h, x:x + w]
              eyes = eye_cascade.detectMultiScale(roi_gray)
              i = 0
              for (ex, ey, ew, eh) in eyes:
                  i += 1
                  cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                  if (i == 2):
                      break
          cv2.imshow('frame', frame)

          k = cv2.waitKey(1) & 0xff
          if k == 27:
              break
      cap.release()
      cv2.destroyAllWindows()
      self.label.SetLabelText("Camera 1 is activated")

    def onBtn2Click(self, event):
        self.label.SetLabelText("Camera 2 is activated")


class MyApp(wx.App):
       def OnInit(self):
        self.frame = MyFrame(parent = None, title="Fatigue Detection")
        self.frame.Show()
        return True


app = MyApp()
app.MainLoop()