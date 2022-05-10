import tkinter as tk
from tkinter import font as tkfont
import cv2
from tkinter import PhotoImage
from PIL import Image, ImageTk
from tkinter.ttk import Label
import pose_analysis_module as pam
from tkinter import ttk


# derived class with parent class as tk.Tk, defines properties of the frame widget with container object or
# parent = container argument and exports self attributes of MainUI class to other classes through controller = self
class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title_font2 = tkfont.Font(family='Helvetica', size=64, weight="bold")
        self.title("Your Exercise Guide")
        self.nontitle_font = tkfont.Font(family='Helvetica', size=16)
        self.geometry("1300x700")
        container = ttk.Frame(self, padding=(0, 0, 0, 0))
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=100)
        container.grid_columnconfigure(0, weight=100)
        self.frames = {}
        for F in (StartPage, PageOne, PageSquat, SquatInstruction1, AnalysisReport):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    # brings the frame in context to the top position in the stack
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


# welcome page
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        render = PhotoImage(file='images/1_midsize.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=2, rowspan=3, sticky="nsew")
        label = tk.Label(self, text="        Home Page        ", font=self.controller.title_font, fg="#263942")
        label.grid(row=0, column=1, rowspan=4, sticky="new", pady=100)
        button1 = tk.Button(self, text="       take pose test      ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("PageOne"))
        button1.grid(row=1, column=1, rowspan=4, pady=10, ipady=4, ipadx=5)
        canv = tk.Canvas(self, bg="black", height=750, width=450)
        canv.create_text(100, 150, text="   YOUR", fill="white", font=self.controller.title_font2)
        # canv.create_text(100, 350, text="     ---------", fill="white", font=self.controller.title_font2)
        canv.create_line(30, 350, 300, 350, fill="white")
        canv.create_text(100, 550, text="      GUIDE.", fill="white", font=self.controller.title_font2)
        canv.grid(row=2, column=3)


# UI pageone
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label1 = tk.Label(self,
                          text="                                                               choose an exercise      "
                               "                          ",
                          font=self.controller.title_font, fg="#263942")
        label1.grid(row=0, column=0, columnspan=7, rowspan=2, sticky="s")
        button1 = tk.Button(self, text="   Squats  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("PageSquat"))
        button1.grid(row=2, column=3, rowspan=2, columnspan=3, pady=10, ipady=4, ipadx=5, sticky="nsew")
        button2 = tk.Button(self, text="   Dead lift  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("Pagedeadlift"))
        button2.grid(row=4, column=3, rowspan=2, columnspan=3, pady=10, ipady=4, ipadx=5, sticky="nsew")
        self.button_cancel = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                       command=lambda: controller.show_frame("StartPage"))
        self.button_cancel.grid(row=6, column=3, rowspan=2, columnspan=3, pady=10, ipady=4, ipadx=5, sticky="nsew")


# page specific to squat exercise to be tested
class PageSquat(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label1 = tk.Label(self,
                          text="                                                                   demo or a test?     "
                               "                           ",
                          font=self.controller.title_font, fg="#263942")
        label1.grid(row=0, column=0, columnspan=7, rowspan=2, sticky="s")
        button_demo = tk.Button(self, text="   play demo  ", fg="#ffffff", bg="#263942",
                                command=lambda: self.demo_video())
        button_demo.grid(row=2, column=3, rowspan=2, columnspan=3, pady=10, ipady=4, ipadx=5, sticky="nsew")

        button_test_squat = tk.Button(self, text="   test your squat   ", fg="#ffffff", bg="#263942",
                                      command=lambda: self.controller.show_frame("SquatInstruction1"))
        button_test_squat.grid(row=4, column=3, rowspan=2, columnspan=3, pady=10, ipady=4, ipadx=5, sticky="nsew")
        self.button_cancel = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                       command=lambda: controller.show_frame("StartPage"))
        self.button_cancel.grid(row=6, column=3, rowspan=2, columnspan=3, pady=10, ipady=4, ipadx=5, sticky="nsew")

    @staticmethod
    # needs to be generalized to access demo videos of varios excersises
    def demo_video():
        cap = cv2.VideoCapture('demovideos\squatdemo.mp4')
        if not cap.isOpened():
            print("Error opening video stream or file")
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
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


# sub-page specific to squat exercise
class SquatInstruction1(tk.Frame):
    def __init__(self, parent, controller):
        front_count = 0
        side_count = 0
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label1 = tk.Label(self, text="Instructions to follow for completing Squat Pose test",
                          font=self.controller.title_font, fg="#263942")
        label1.grid(row=0, column=0, columnspan=7, rowspan=2, sticky="nsew")
        label2 = tk.Label(self, text=" You see two demonstrations below and their respective 'Next' buttons.",
                          font=self.controller.nontitle_font, fg="#263942")
        label2.grid(row=3, column=0, columnspan=7, sticky="ew")
        label3 = tk.Label(self, text="Click Next1 and distance yourself until you fit inside your laptop screen... "
                                     "correct symbol should appear...",
                          font=self.controller.nontitle_font, fg="#263942")
        label3.grid(row=4, column=0, columnspan=7, sticky="ew")
        label4 = tk.Label(self, text="Follow the Instruction on the screen.",
                          font=self.controller.nontitle_font, fg="#263942")
        label4.grid(row=5, column=0, columnspan=7, sticky="ew")
        label5 = tk.Label(self,
                          text=" Repeat the same with Next2 button.",
                          font=self.controller.nontitle_font, fg="#263942")
        label5.grid(row=6, column=0, columnspan=7)
        label6 = tk.Label(self,
                          text=" Finally Click 'Generate report' to access the analysis report.",
                          font=self.controller.nontitle_font, fg="#263942")
        label6.grid(row=7, column=0, columnspan=7)
        image_fit = Image.open("images/front_2_small.png")
        fit_screen = ImageTk.PhotoImage(image_fit)
        labelimg1 = Label(self, image=fit_screen)
        labelimg1.image = fit_screen

        image_fit_side = Image.open("images/side_2_small.png")
        fit_screen_s = ImageTk.PhotoImage(image_fit_side)
        labelimg2 = Label(self, image=fit_screen_s)
        labelimg2.image = fit_screen_s
        labelimg1.grid(column=0, row=8, rowspan=5, columnspan=3)
        labelimg2.grid(column=4, row=8, rowspan=5, columnspan=3)

        pam_obj = pam.PoseAnalysisOperations()
        button_next1 = tk.Button(self, text="Next1",
                                 command=lambda: pam_obj.fit_posekeyp_inscreen_check(front_count, side_count,
                                                                                     front=True), fg="#ffffff",
                                 bg="#263942")
        button_next1.grid(row=13, column=1, ipadx=5, ipady=4, pady=10)
        button_next2 = tk.Button(self, text="Next2",
                                 command=lambda: pam_obj.fit_posekeyp_inscreen_check(front_count, side_count,
                                                                                     front=False),
                                 fg="#ffffff", bg="#263942")
        button_next2.grid(row=13, column=5, ipadx=5, ipady=4, pady=10)
        button_next3 = tk.Button(self, text="Generate Report", command=lambda: controller.show_frame("AnalysisReport"),
                                 fg="#ffffff", bg="#263942")
        button_next3.grid(row=9, column=7, ipadx=20, ipady=20, pady=20)
        self.button_cancel = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                       command=lambda: controller.show_frame("StartPage"))
        self.button_cancel.grid(row=11, column=7, ipadx=40, ipady=20, pady=20)


class AnalysisReport(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label1 = tk.Label(self, text="                                                  "
                                     "Your Pose Analysis Report.",
                          font=self.controller.title_font, fg="#263942")
        label1.grid(row=0, column=0, columnspan=7, rowspan=2, sticky="nsew")
        self.button_cancel = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                       command=lambda: controller.show_frame("StartPage"))
        self.button_cancel.grid(row=11, column=5, ipadx=40, ipady=20, pady=200, padx=10)


app = MainUI()
app.mainloop()
