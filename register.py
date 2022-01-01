import os
import sqlite3
from tkinter import *
from tkinter import messagebox, ttk

from PIL import ImageTk

root = Tk()
root.geometry("1300x900")
root.title("registration form")

Email = StringVar()
Password = StringVar()
FIRSTNAME = StringVar()
LASTNAME = StringVar()


def Database():
    global conn, cursor
    conn = sqlite3.connect("db_member.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS members (mem_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, username TEXT, password TEXT, firstname TEXT, lastname TEXT)")


def Register():
    Database()
    if Email.get == "" or Password.get() == "" or FIRSTNAME.get() == "" or LASTNAME.get == "":
        print("Hii")
    else:
        cursor.execute("SELECT * FROM members WHERE username = ?", (Email.get(),))
        if cursor.fetchone() is not None:
            print("Hii")
        else:
            cursor.execute("INSERT INTO members (username, password, firstname, lastname) VALUES(?, ?, ?, ?)",
                           (str(Email.get()), str(Password.get()), str(FIRSTNAME.get()), str(LASTNAME.get())))
            conn.commit()
            Email.set("")
            Password.set("")
            FIRSTNAME.set("")
            LASTNAME.set("")
            os.system('python login.py')
            root.destroy()
        cursor.close()
        conn.close()


bg = ImageTk.PhotoImage(file="img2.png")
label5 = Label(root, image=bg)
label5.place(x=0, y=0)

label_1 = Label(root, text="Registration Form", fg="white", bg="#660066", borderwidth=10, width=15, height=1,
                activebackground="white", font=('times', 20, ' bold '))
# place method in tkinter is  geometry manager it is used to organize widgets by placing them in specific position
label_1.place(x=550, y=40)

lbl1 = Label(root, text="Email", width=18, height=2, fg="Black", bg="#ffe6e6", font=('times', 18, ' bold '))
lbl1.place(x=400, y=200)

username = Entry(root, width=20, bg="#e1fcff", textvariable=Email, fg="black", font=('times', 17, ' bold '))
username.place(x=700, y=200, width=250, height=50)

lbl2 = Label(root, text="Password", width=18, height=2, fg="black", bg="#ffe6e6", font=('times', 18, ' bold '))
lbl2.place(x=400, y=300)

password = Entry(root, width=20, bg="#e1fcff", textvariable=Password, fg="black", font=('times', 17, ' bold '),show="*")
password.place(x=700, y=300, width=250, height=50)

lbl2 = Label(root, text="First Name", width=18, fg="black", bg="#ffe6e6", height=2, font=('times', 18, ' bold '))
lbl2.place(x=400, y=400)

firstname = Entry(root, width=20, bg="#e1fcff", textvariable=FIRSTNAME, fg="black", font=('times', 17, ' bold '))
firstname.place(x=700, y=400, width=250, height=50)

lbl3 = Label(root, text="Last Name", width=18, fg="black", bg="#ffe6e6", height=2, font=('times', 18, ' bold '))
lbl3.place(x=400, y=500)

lastname = Entry(root, width=20, bg="#e1fcff", textvariable=LASTNAME, fg="black", font=('times', 17, ' bold '))
lastname.place(x=700, y=500, width=250, height=50)



register = Button(root, text="Register", fg="white", command=Register, bg="#660066", borderwidth=10, width=18, height=2,
                  activebackground="Red", font=('times', 15, ' bold '))
register.place(x=580, y=650)

root.mainloop()
