from tkinter import *
from tkinter import ttk

root = Tk()

content = ttk.Frame(root)
frame = ttk.Frame(content, borderwidth=5, relief="ridge", width=1500, height=650)

content.grid(column=0, row=0)
frame.grid(column=0, row=0, columnspan=3, rowspan=2)

root.mainloop()