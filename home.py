import os
from tkinter import *
from tkinter import messagebox

from PIL import ImageTk

root=Tk()
root.geometry("1300x900")
root.title("home")


def Logout():
    messagebox.showinfo("Log Out", "Logged Out Successfully")
    root.destroy()
    os.system('python login.py')
def face():
    os.system('python Capstone_2.0.py')




bg= ImageTk.PhotoImage(file="img2.png")
label5=Label(root,image=bg)
label5.place(x=0,y=0)

menubar = Menu(root)
report = Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='Report', menu = report)
report.add_command(label ='View Report')

exit= Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='Exit', menu = exit)
exit.add_command(label ='Log Out', command = Logout)



label_1 =Label(root, text="Intelligence System For Personality Assessment " ,fg="black"  ,bg="#ffccdd",borderwidth =15,width=40  ,height=1, activebackground = "white" ,font=('times', 27, ' bold '))
#place method in tkinter is  geometry manager it is used to organize widgets by placing them in specific position
label_1.place(x=250,y=60)



take_asses= Button(root, text="Take Assesment" ,fg="black",bg="#b3ffb3",borderwidth = 25,width=15  ,height=2, activebackground = "Red" ,font=('times', 28, ' bold '),command=face)
take_asses.place(x=460, y=350)



root.config(menu = menubar)
root.mainloop()