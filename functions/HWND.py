
import cv2 as cv2
import numpy as np
# from PIL import ImageGrab
import pyautogui
import time
from mss import mss
from PIL import Image
import threading
import time
import pywintypes # // https://stackoverflow.com/questions/3956178/cant-load-pywin32-library-win32gui adding for import dll file to init win32gui
import win32gui, win32ui, win32api, win32con
from ctypes import windll


class Script():
 
    def __init__(self):
        global hWnd 
        # hWnd =  win32gui.FindWindow(None, 'Calculator')
        # hWnd =  win32gui.FindWindow(None, 'LINE')
        hWnd =  win32gui.FindWindow(None, 'RF Online')
        
        return
 
    def Screen_Capture(self):
        
        hwnd = hWnd
        # print(hWnd)
        # for i in range(0,100):
        #     self.Mouse_movement(500+i,500+i)
        # hwnd = win32gui.FindWindow(None, 'Calculator')
        # https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui
        
        # Uncomment the following line if you use a high DPI display or >100% scaling size
        # windll.user32.SetProcessDPIAware()

        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #left, top, right, bot = win32gui.GetClientRect(hwnd)
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
       

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        if result == 1:
            #PrintWindow Succeeded
            im.save("./data/0001.jpg")
        
 
    def Show_Screen(self):
        self.Screen_Capture()
        
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        while (True):
            cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
            ret, frame = cap.read()
            gray = object_detector.apply(frame) 
            _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            dilated = cv2.dilate(mask_eroded,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)
    
            contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
            min_contour_area = 10  # Define your minimum area threshold
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            frame_out = frame.copy()
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = (x*2 + w)/2
                center_y = (y*2 + h)/2
                self.Mouse_movement(center_x,center_y)
                # self.Mouse_movement(center_x,center_y)
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            self.Screen_Capture()
    
    def Mouse_movement(self,x,y):
        x,y = int(x),int(y) 
        hWnd1= win32gui.FindWindowEx(hWnd, None, None, None)
        lParam = win32api.MAKELONG(x,y)
        pyautogui.leftClick(x, y)
        win32api.SendMessage(hWnd1, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
        win32api.SendMessage(hWnd1, win32con.WM_LBUTTONUP, None, lParam)

        

    def Object_detection (self):
        
        return 


script = Script()
script.Screen_Capture()
script.Show_Screen()

 
 