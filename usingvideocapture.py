
import cv2 as cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
from mss import mss
from PIL import Image
import threading
import time
import pywintypes # // https://stackoverflow.com/questions/3956178/cant-load-pywin32-library-win32gui adding for import dll file to init win32gui
import win32gui
import pydirectinput
import gc


class Script():
 
    def __init__(self):
        self.template = cv2.imread("./template/template.jpg")
        return
 
    def Screen_Capture(self):
        bounding_box = {'top': 1, 'left': 1, 'width': 1300, 'height': 800}
        i = 0
        sct = mss()
        while (i < 1):
            screenshot = np.array(sct.grab(bounding_box))
            # screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            a = cv2.imwrite("./data/0001.jpg".format(i),screenshot)            
            i = i + 1
            del screenshot
            gc.collect()
    
    def Show_Screen(self):
        self.Screen_Capture()
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        while (True):
            cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
            ret, frame = cap.read()
            frame_out = frame.copy()
            gray = object_detector.apply(frame) 
            _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            dilated = cv2.dilate(mask_eroded,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)
    
            contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
            min_contour_area = 500  # Define your minimum area threshold
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = (x*2 + w)/2
                center_y = (y*2 + h)/2
                # self.Mouse_movement(center_x,center_y)
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
                # check if find monster
            result = self.If_clickMonster(frame_out,self.template)
            print(result)
            cv2.imshow('frame', frame_out)
            if cv2.waitKey(1) == ord('q'):
                break


            self.Screen_Capture()
    
    def Mouse_movement(self,x,y):
        x,y = int(x),int(y)
        pydirectinput.moveTo(x, y)
        pydirectinput.click()
        return 
    def If_clickMonster(self,frame,template):
        template = cv2.imread("./template/template.jpg")
        frame = cv2.imread("./template/0001.jpg")
        
        resize_template = template#cv2.resize(template,(frame.shape[1],frame.shape[0])) 
        res = (cv2.matchTemplate(frame,resize_template,cv2.TM_CCOEFF_NORMED))
        # cv2.imwrite("./template/0002.jpg",resize_template)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        return maxVal
    
    



script = Script()
script.Screen_Capture()
script.Show_Screen()

 
 