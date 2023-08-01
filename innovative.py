import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from tkinter import *
from win32api import GetSystemMetrics
import os
import cv2
import numpy as np


def browse_img():
    filename = filedialog.askopenfilename(initialdir="Desktop", title="Select Image File",
                                          filetypes=(
                                          ("JPEG files", "*.jpeg"), ("JPG files", "*.jpg"), ("PNG files", "*.png"),
                                          ("GIF files", "*.gif"), ("ICON files", "*.ico"), ("All Files", "*.*")))
    return filename

def image_resize():
    top = tk.Toplevel()
    top.title("Image resizing")

    color_frame = tk.LabelFrame(top, text="Options", borderwidth=3)
    color_frame.pack(fill="both", expand="yes", padx=10, pady=10)
    
    scale_entry=ttk.Entry(color_frame)
    scale_entry.grid(row=0, column=0, padx=10, pady=10)
    
    s=scale_entry.get()
    scale_entry.delete(0,END)

    by_scale = ttk.Button(color_frame, text="By scale", command=lambda:resize_by_scale(int(s)))
    by_scale.grid(row=0, column=1, padx=10, pady=10)

    width_entry=ttk.Entry(color_frame)
    width_entry.grid(row=1, column=0, padx=10, pady=10)

    w=width_entry.get()
    width_entry.delete(0,END)

    by_width = ttk.Button(color_frame, text="By width", command=lambda:resize_by_width(w))
    by_width.grid(row=1, column=1, padx=10, pady=10)

    height_entry=ttk.Entry(color_frame)
    height_entry.grid(row=2, column=0, padx=10, pady=10)

    h=height_entry.get()
    height_entry.delete(0,END)

    by_height= ttk.Button(color_frame, text="By height", command=lambda:resize_by_height(h))
    by_height.grid(row=2, column=1, padx=10, pady=10)


def resize_by_scale(img,scale=0.75):
    if img=='':
        filename=browse_img()
        img=cv2.imread(filename)

    h = int(img.shape[1] * scale)
    w = int(img.shape[0] * scale)
    d = (h,  w)
    img1=cv2.resize(img, d, interpolation = cv2.INTER_AREA)
    cv2.imshow('img',img)
    cv2.imshow('img1',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_by_width(width = GetSystemMetrics(0) / 2):
    filename=browse_img()
    img=cv2.imread(filename)
    org_w = int(img.shape[0])
    scale = width / org_w
    return resize_by_scale(img, cv2, scale = scale)


def resize_by_height(height = GetSystemMetrics(1) / 2):
    filename=browse_img()
    img=cv2.imread(filename)
    org_h = int(img.shape[1])
    scale = int(float(height)) / org_h
    return resize_by_scale(img, cv2, scale = scale)

def thresholding():
    filename=browse_img()
    img=cv2.imread(filename)
    retval, threshold=cv2.threshold(img,12,255,cv2.THRESH_BINARY)

    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    retval, threshold=cv2.threshold(gray_img,12,255,cv2.THRESH_BINARY)
    gaus=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
    retval12,otsu=cv2.threshold(gray_img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('original',img)
    cv2.imshow('threshold',threshold)
    cv2.imshow('gray_img',gray_img)
    cv2.imshow('gaus',gaus)
    cv2.imshow('otsu',otsu)
    print(retval)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def live_video_filtering():
    top = tk.Toplevel()
    top.title("Video filtering")

    color_frame = tk.LabelFrame(top, text="Color Options", borderwidth=3)
    color_frame.pack(fill="both", expand="yes", padx=10, pady=10)

    gray_button = ttk.Button(color_frame, text="Blue", command=lambda:video_filtering([60, 35, 140],[180,255,255]))
    gray_button.grid(row=0, column=0, padx=10, pady=10)

    hsv_button = ttk.Button(color_frame, text="Red", command=lambda:video_filtering([30,150,50],[255,255,180]))
    hsv_button.grid(row=1, column=0, padx=10, pady=10)

    lab_button = ttk.Button(color_frame, text="Yellow", command=lambda:video_filtering([20, 100, 100],[30, 255, 255]))
    lab_button.grid(row=0, column=1, padx=10, pady=10)

    luv_button = ttk.Button(color_frame, text="Orange", command=lambda:video_filtering([10, 100, 20],[25, 255, 255]))
    luv_button.grid(row=1, column=1, padx=10, pady=10)



def video_filtering(lower_r,upper_r):
    cap=cv2.VideoCapture(0)

    while True:
        _,frame=cap.read()
        hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lower_red=np.array(lower_r)
        upper_red=np.array(upper_r)

        # dark_red=np.uint8([[[12,22,121]]])
        # dark_red=cv2.cvtColor(dark_red,cv2.COLOR_BGR2HSV)

        mask=cv2.inRange(hsv,lower_red,upper_red)    
        res=cv2.bitwise_and(frame,frame,mask=mask)

        cv2.imshow('frame',frame)
        # cv2.imshow('mask',mask)
        cv2.imshow('res',res)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


def motion_detection(url):
    cap = cv2.VideoCapture(url)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # print(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thrash = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thrash, None, iterations = 3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 500:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()



def facRec():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # cv.imshow('frame', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y ,w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        # print(faces)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray2, 1.1, 4)
        for (x, y ,w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()



def color_filter():
    top = tk.Toplevel()
    top.title("Change Image Color")

    color_frame = tk.LabelFrame(top, text="Color Options", borderwidth=2)
    color_frame.pack(fill="both", expand="yes", padx=10, pady=10)

    gray_button = ttk.Button(color_frame, text="GRAY", command=lambda:filter(1))
    gray_button.grid(row=0, column=0, padx=10, pady=10)

    hsv_button = ttk.Button(color_frame, text="HSV", command=lambda:filter(2))
    hsv_button.grid(row=1, column=0, padx=10, pady=10)

    lab_button = ttk.Button(color_frame, text="LAB", command=lambda:filter(3))
    lab_button.grid(row=0, column=1, padx=10, pady=10)

    luv_button = ttk.Button(color_frame, text="LUV", command=lambda:filter(4))
    luv_button.grid(row=1, column=1, padx=10, pady=10)

    hls_button = ttk.Button(color_frame, text="HLS", command=lambda:filter(5))
    hls_button.grid(row=2, column=0, padx=10, pady=10)

    yuv_button = ttk.Button(color_frame, text="YUV", command=lambda:filter(6))
    yuv_button.grid(row=2, column=1, padx=10, pady=10)

def edge_detection():
    cap=cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        lower_red = np.array([30,150,50])
        upper_red = np.array([255,255,180])
    
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('Original',frame)
        edges = cv2.Canny(frame,100,200)
        cv2.imshow('Edges',edges)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.release()


def filter(number):
    filename=browse_img()
    img=cv2.imread(filename)
    if number==1:
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif number==2:
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif number==3:
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif number==4:
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif number==5:
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif number==6:
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    cv2.imshow('cv_img',cv_img)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
if __name__=='__main__':

    root=tk.Tk()

    heading=tk.Frame(root,borderwidth=5,highlightbackground="blue", highlightthickness=2)
    heading.grid(row=0,column=0,padx=10, pady=10)

    frame=tk.Frame(root,borderwidth=5,highlightbackground="black", highlightthickness=2)
    frame.grid(row=1,column=0,padx=10, pady=10)

    frame1=tk.Frame(frame,borderwidth=5,highlightbackground="red", highlightthickness=2)
    frame1.grid(row=2,column=0,padx=10, pady=10)

    frame2=tk.Frame(frame,borderwidth=5,highlightbackground="yellow", highlightthickness=2)
    frame2.grid(row=2,column=12,padx=10, pady=10)

    # color_frame = tk.LabelFrame(top, text="Color Options", borderwidth=2)
    # color_frame.pack(fill="both", expand="yes", padx=10, pady=10)

    label1=tk.Label(heading,text='Image and Video Processing',width=40,height=2,font=("Arial", 15))
    label1.grid(row=0,column=0,padx=10, pady=10)

    label2=tk.Label(frame1,text='Image',font=("Arial", 20))
    label2.grid(row=3,column=0,padx=6,pady=6)

    button11=tk.Button(frame1,text='Resize',command=image_resize,borderwidth=5,bg='white',width=25)
    button11.grid(row=4,column=0,padx=6,pady=6)

    button12=tk.Button(frame1,text='Filter',command=color_filter,borderwidth=5,bg='white',width=25)
    button12.grid(row=5,column=0,padx=6,pady=6)

    button13=tk.Button(frame1,text='Thresholding',command=thresholding,borderwidth=5,bg='white',width=25)
    button13.grid(row=6,column=0,padx=6,pady=6)
    


    label3=tk.Label(frame2,text='Video',font=("Arial", 20))
    label3.grid(row=3,column=12,padx=6,pady=6)

    button21=tk.Button(frame2,text='Facial feature detection',command=facRec,borderwidth=5,bg='white',width=25)
    button21.grid(row=4,column=12,padx=6,pady=6)

    button22=tk.Button(frame2,text='Motion Detection',command=lambda:motion_detection("ghost.mp4"),borderwidth=5,bg='white',width=25)
    button22.grid(row=5,column=12,padx=6,pady=6)

    button23=tk.Button(frame2,text='Edge detection',command=edge_detection,borderwidth=5,bg='white',width=25)
    button23.grid(row=6,column=12,padx=6,pady=6)
    
    button24=tk.Button(frame2,text='Color filtering',command=live_video_filtering,borderwidth=5,bg='white',width=25)
    button24.grid(row=7,column=12,padx=6,pady=6)

    root.mainloop()

