from tkinter import*
import tkinter as tk
import random
import time
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.geometry("1600x700+0+0")
root.title("Stock Analysis")
root.configure(background= "#BC8F8F")

Tops = Frame(root,bg="white",width = 1600,height=50,relief=SUNKEN)
Tops.pack(side=TOP)

f1 = Frame(root,width = 900,height=700,relief=SUNKEN)
f1.pack(side=LEFT)

f2 = Frame(root ,width = 750,height=700,relief=SUNKEN,background="#CD5C5C")
f2.pack(side=RIGHT)

#------------------TIME--------------
localtime=time.asctime(time.localtime(time.time()))
#-----------------INFO TOP------------
lblinfo = Label(Tops, font=( 'aria' ,30, 'bold' ),text="Stock Market Statistical & Sentimental Analysis",fg="#4169E1",bd=10,anchor='w')
lblinfo.grid(row=0,column=0)
lblinfo = Label(Tops, font=( 'aria' ,20, ),text=localtime,fg="steel blue",anchor=W)
lblinfo.grid(row=1,column=0)

def qexit():
    root.destroy()
def reset():
    txtapple.set("")
    Fries.set("")
    Largefries.set("")
    Burger.set("")
'''
status = Label(f2,font=('aria', 15, 'bold'),width = 16, text="-By Team Stonks",bd=2,relief=SUNKEN)
status.grid(row=0,columnspan=1)
'''
def applesenti():
    pass
def amazonsenti():
    pass
def facebooksenti():
    pass
def gssenti():
    pass
def googlesenti():
    pass
def intelsenti():
    pass
def microsenti():
    pass



btnc=Button(f1,padx=16,pady=16,bd=4, fg="black", font=('ariel', 20 ,'bold'),text="Apple",bg="powder blue", command=applesenti)
btnc.grid(row=0,column=0)
txtapple = Entry(f1,font=('ariel' ,16,'bold'), textvariable="" , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
txtapple.grid(row=0,column=1)

btnc=Button(f1,padx=16,pady=16,bd=4, fg="black", font=('ariel', 20 ,'bold'),text="Amazon",bg="powder blue", command=applesenti)
btnc.grid(row=1,column=0)
txtamz = Entry(f1,font=('ariel' ,16,'bold'), textvariable="" , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
txtamz.grid(row=1,column=1)

btnc=Button(f1,padx=16,pady=16,bd=4, fg="black", font=('ariel', 20 ,'bold'),text="Facebook",bg="powder blue", command=applesenti)
btnc.grid(row=2,column=0)
txtfb = Entry(f1,font=('ariel' ,16,'bold'), textvariable="" , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
txtfb.grid(row=2,column=1)


btnc=Button(f1,padx=16,pady=16,bd=4, fg="black", font=('ariel', 20 ,'bold'),text="Goldman Sachs",bg="powder blue", command=applesenti)
btnc.grid(row=3,column=0)
txtgs = Entry(f1,font=('ariel' ,16,'bold'), textvariable="" , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
txtgs.grid(row=3,column=1)

btnc=Button(f1,padx=16,pady=16,bd=4, fg="black", font=('ariel', 20 ,'bold'),text="Google",bg="powder blue", command=applesenti)
btnc.grid(row=4,column=0)
txtgg = Entry(f1,font=('ariel' ,16,'bold'), textvariable="" , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
txtgg.grid(row=4,column=1)

btnc=Button(f1,padx=16,pady=16,bd=4, fg="black", font=('ariel', 20 ,'bold'),text="Intel",bg="powder blue", command=applesenti)
btnc.grid(row=5,column=0)
txtintel = Entry(f1,font=('ariel' ,16,'bold'), textvariable="" , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
txtintel.grid(row=5,column=1)

btnc=Button(f1,padx=16,pady=16,bd=4, fg="black", font=('ariel', 20 ,'bold'),text="Microsoft",bg="powder blue", command=applesenti)
btnc.grid(row=6,column=0)
txtmicro = Entry(f1,font=('ariel' ,16,'bold'), textvariable="" , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
txtmicro.grid(row=6,column=1)

btnreset=Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=10, text="RESET", bg="powder blue",command=reset)
btnreset.grid(row=7, column=2)


btnexit=Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=10, text="EXIT", bg="powder blue",command=qexit)
btnexit.grid(row=7, column=3)
#-------------------------------------------------------

data3 = {'Company': ['Apple', 'Amazon', 'Intel', 'Microsoft', 'Goldman Sachs'],
         'Stock_Index_Price': [1500, 1520, 1525, 1523, 1515]  }
df3 = DataFrame(data3, columns=['Company', 'Stock_Index_Price'])

figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df3['Company'], df3['Stock_Index_Price'], color='g')
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['Stock_Index_Price'])
ax3.set_xlabel('Company')
ax3.set_title('Statistical Analysis')
#-------------------------------------------------------


root.mainloop()
