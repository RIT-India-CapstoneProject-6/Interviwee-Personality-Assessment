import os
import sqlite3
from tkinter import *

from PIL import ImageTk

root=Tk()
root.geometry("1300x900")
root.title("login form")


Email = StringVar()
Password = StringVar()



def Database():
    global conn, cursor
    conn = sqlite3.connect("db_member.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS members (mem_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, username TEXT, password TEXT, firstname TEXT, lastname TEXT)")




def RegisterFrame():
    root.destroy()
    os.system('python register.py')

def LogIn():
    Database()
    if Email.get == "" or Password.get() == "":
        print("Please complete the required field!")
    else:
        cursor.execute("SELECT * FROM members WHERE  username = ? and password = ?",
                       (Email.get(), Password.get()))
        if cursor.fetchone() is not None:
            print("You Successfully Login")
            os.system('python home.py')
        else:
            print("Invalid Username or password")


bg= ImageTk.PhotoImage(file="img2.png")
label5=Label(root,image=bg)
label5.place(x=0,y=0)

label_1 =Label(root, text="LogIn Form" ,fg="white"  ,bg="#660066",borderwidth = 10,width=15  ,height=1, activebackground = "white" ,font=('times', 20, ' bold '))
#place method in tkinter is  geometry manager it is used to organize widgets by placing them in specific position
label_1.place(x=550,y=40)

lbl1 = Label(root, text="Enter Email",width=18  ,height=2  ,fg="black"  ,bg="#ffe6e6" ,font=('times', 18, ' bold ') )
lbl1.place(x=400, y=200)

username = Entry(root,width=20   ,bg="#e1fcff" ,fg="black",font=('times', 17, ' bold '),textvariable=Email)
username.place(x=700, y=200,width=250,height=50)

lbl2 = Label(root, text="Enter Password",width=18  ,height=2  ,fg="black"  ,bg="#ffe6e6" ,font=('times', 18, ' bold ') )
lbl2.place(x=400, y=400)

password = Entry(root,width=20   ,bg="#e1fcff" ,fg="black",font=('times', 17, ' bold '),textvariable=Password,show="*")
password.place(x=700, y=400,width=250,height=50)



register= Button(root, text="Register" ,fg="white",command=RegisterFrame  ,bg="#660066",borderwidth = 10,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
register.place(x=400, y=600)

login= Button(root, text="LogIn" ,fg="white",command=LogIn  ,bg="#660066",borderwidth = 10,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
login.place(x=700, y=600)


root.mainloop()