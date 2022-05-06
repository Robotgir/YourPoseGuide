import os
import sys
import tkinter as tk
from tkinter import font as tkfont
import cv2
from tkinter import messagebox, PhotoImage
import PIL.Image, PIL.ImageTk
import time
import PoseModule as pm
from PIL import Image, ImageTk
from tkinter.ttk import Frame, Label, Style
import HolisticModule as hm
from tkinter import ttk
import math
count=0
scount=0
global a
global b
class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Your Pose Guide")
        self.nontitle_font = tkfont.Font(family='Helvetica', size=16)
        self.resizable(False, False)
        self.geometry("1300x700")
        # self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = ttk.Frame(self, padding=(0,0,300,300))
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, Pagesquat, Instruction1):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        render = PhotoImage(file='images/1.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=1, rowspan=4, sticky="nsew")
        label = tk.Label(self, text="        Home Page        ", font=self.controller.title_font, fg="#263942")
        label.grid(row=0, sticky="ew")
        button1 = tk.Button(self, text="   take basic poses test  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("Instruction1"))
        button1.grid(row=1, column=0, ipady=3, ipadx=7)


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="        choose an exercise        ", font=self.controller.title_font, fg="#263942")
        label.grid(row=0, sticky="ew")
        button1 = tk.Button(self, text="   Squats  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("Pagesquat"))
        button2 = tk.Button(self, text="   Dead lift  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("Pagedeadlift"))
        button1.grid(row=1, column=0, ipady=3, ipadx=7)
        button2.grid(row=2, column=0, ipady=3, ipadx=2)
        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttoncanc.grid(row=10, column=0, pady=10, ipadx=5, ipady=4)


class Pagesquat(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        buttondemo = tk.Button(self, text="play demo", fg="#ffffff", bg="#263942",
                                    command=lambda: self.demovideo())
        buttontest_squat = tk.Button(self, text="test your squat", fg="#ffffff", bg="#263942",
                                          command=lambda: self.controller.show_frame("Instruction1"))
        buttondemo.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)
        buttontest_squat.grid(row=2, column=1, pady=10, ipadx=5, ipady=4)

'''
class Instruction1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label1 = tk.Label(self, text="        Instructions to follow for completing Squat test       ",
                         font=self.controller.title_font, fg="#263942")
        label1.grid(row=0, sticky="ew")
        label2 = tk.Label(self, text="Click next and distance yourself from the camera until your whole body fits in the laptop screen as shown in below images respectively ...",
                         font=self.controller.nontitle_font, fg="#263942")
        label2.grid(row=2, column=0, columnspan=3, sticky="ew")
        label3 = tk.Label(self,
                         text="Once the correct symbol appears  on the screen the countdown of 5 seconds starts.",
                         font=self.controller.nontitle_font, fg="#263942")
        label3.grid(row=3,column=0, columnspan=3,  sticky="ew")
        label4 = tk.Label(self,
                         text="After the countdown perform 3 Squats, recording will automatically stop after three squats ...",
                         font=self.controller.nontitle_font, fg="#263942")
        label4.grid(row=4, sticky="ew")
        label5 = tk.Label(self,
                         text=" Repeat the above instructions for capturing front and side poses ",
                         font=self.controller.nontitle_font, fg="#263942")
        label5.grid(row=5, sticky="ew")
        imagefit = Image.open("images/front_1_small.png")
        fitscreen= ImageTk.PhotoImage(imagefit)
        labelimg1 = Label(self, image=fitscreen)
        labelimg1.image = fitscreen

        imagefit_side = Image.open("images/side_realhuman.jpg")
        fitscreen_s = ImageTk.PhotoImage(imagefit_side)
        labelimg2 = Label(self, image=fitscreen_s)
        labelimg2.image = fitscreen_s

        #img = tk.Label(self, image=render)
        #img.image = render
        #img.grid(row=2, column=1, rowspan=4, sticky="nsew")
        #label.place(x=270,y=300)
        labelimg1.grid(column=0, row=7, rowspan=5, columnspan=3)
        labelimg2.grid(column=4, row=7, rowspan=5, columnspan=3)
        buttonext = tk.Button(self, text="Next", command=self.holisticDetection, fg="#ffffff", bg="#263942")
        buttonext.grid(row=8,column=0, ipadx=5, ipady=4, pady=10)

    def demovideo(self):
        cap = cv2.VideoCapture('demovideos\squat.mp4')
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

    def posedetection(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        pTime = 0
        detector = pm.poseDetector()
        while True:
            success, img = cap.read()
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                print(lmList[14])
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

    def holisticDetection(self):
        holisticobj = hm.holistic()
        holisticobj.findHolistic(video_source=0)

    def nextfoo(self):
        self.holisticDetection()

'''

class Instruction1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label1 = tk.Label(self, text="Instructions to follow for completing Squat test",
                         font=self.controller.title_font, fg="#263942")
        label1.grid(row=0, column =0, columnspan =7, rowspan=2, sticky="nsew")
        label2 = tk.Label(self, text="Click next and distance yourself until you fit inside your laptop screen...",
                         font=self.controller.nontitle_font, fg="#263942")
        label2.grid(row=3, column=0, columnspan=7, sticky="ew")
        label3 = tk.Label(self,text="Saw correct symbol ? show a thumbs up! countdown for 5 starts now !!",
                         font=self.controller.nontitle_font, fg="#263942")
        label3.grid(row=4,column=0, columnspan=7,  sticky="ew")
        label4 = tk.Label(self,
                         text="do 3 x squat, facing the cam",
                         font=self.controller.nontitle_font, fg="#263942")
        label4.grid(row=5,column=0,columnspan=7)
        label5 = tk.Label(self,
                         text=" Repeat 3 x squat with your side facing the cam",
                         font=self.controller.nontitle_font, fg="#263942")
        label5.grid(row=6,column=0,columnspan=7)
        imagefit = Image.open("images/front_2_small.png")
        fitscreen= ImageTk.PhotoImage(imagefit)
        labelimg1 = Label(self, image=fitscreen)
        labelimg1.image = fitscreen

        imagefit_side = Image.open("images/side_2_small.png")
        fitscreen_s = ImageTk.PhotoImage(imagefit_side)
        labelimg2 = Label(self, image=fitscreen_s)
        labelimg2.image = fitscreen_s

        #img = tk.Label(self, image=render)
        #img.image = render
        #img.grid(row=2, column=1, rowspan=4, sticky="nsew")
        #label.place(x=270,y=300)
        labelimg1.grid(column=0, row=7, rowspan=5, columnspan=3)
        labelimg2.grid(column=4, row=7, rowspan=5, columnspan=3)

        holisticObj=hm.holisticDetector()
        buttonext1 = tk.Button(self, text="Next", command=lambda: holisticObj.fitPosekeypinScreenCheck(count,scount,front=True), fg="#ffffff", bg="#263942")
        buttonext1.grid(row=12,column=1, ipadx=5, ipady=4, pady=10)
        buttonext2 = tk.Button(self, text="Next", command=lambda: holisticObj.fitPosekeypinScreenCheck(count, scount,front=False), fg="#ffffff", bg="#263942")
        buttonext2.grid(row=12, column=5, ipadx=5, ipady=4, pady=10)

   # def allKeypointsInscreenCheck(self):
        #holisticdetection and other tasks have to run parallely to check if all keypoints are in thelaptop screen



    def demovideo(self):
        cap = cv2.VideoCapture('demovideos\squat.mp4')
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

    def posedetection(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        pTime = 0
        detector = pm.poseDetector()
        while True:
            success, img = cap.read()
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                print(lmList[14])
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

    def allPoseKPinframe_check(self,video_source):
        holisticobj = hm.holisticDetector
        kp=holisticobj.numKeypoints(video_source)
        print(kp)


    def nextfoo(self):
        self.holisticDetection()


class App:
    def __init__(self, window, window_title, video_source='demovideos\squat.mp4'):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source='demovideos\squat.mp4'):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        global ret
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return ret, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


app = MainUI()
app.mainloop()
