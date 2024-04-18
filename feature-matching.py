
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
import pyautogui
import matplotlib.pyplot as plt


class Script():
 
    def __init__(self):
        self.template = cv2.imread("./template/template1.jpg")
        return
 
    def Screen_Capture(self):
        with mss() as sct :
            bounding_box = {'top': 1, 'left': 1, 'width': 1300, 'height': 800}
            screenshot = np.array(sct.grab(bounding_box))
            cv2.imwrite("./data/0001.jpg",screenshot) 
        # https://stackoverflow.com/questions/54719730/when-taking-many-screenshots-with-mss-memory-fills-quickly-and-crashes-python
        # context manager
        return
    
    def Show_Screen(self):
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        while (True):
            self.Screen_Capture()
            cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
            ret, frame = cap.read()
            frame_out = frame
            gray = object_detector.apply(frame) 
            _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            dilated = cv2.dilate(mask_eroded,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)
    
            contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
            min_contour_area = 500  # Define your minimum area threshold
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            center_x_click,center_y_click = 0,0
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = (x*2 + w)/2
                center_y = (y*2 + h)/2
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
                center_x_click,center_y_click = center_x, center_y

            if_clickMonster = self.If_clickMonster(frame,self.template)
            if (if_clickMonster == False) : # check if not seclected
                # print("1. Monster not selected")
                if (len(large_contours)<10) : # wait until screen stationary
                    # print("2. Screen is stationary")
                    self.Mouse_movement(center_x_click,center_y_click)
                else : pass # not seleted and not stationary, pass and wait for next frame
            else : 
                print("we found monster !")
                pass
                 # keep defeating prcoess until not selected, we cant use while beacause we need to reresh frame
                    


                    # try to click monster
                # self.Mouse_movement(center_x,center_y)
                #     while (self.If_clickMonster<200): # check if seclected
                #         self.Process_of_defeat()
                # self.Mouse_movement(center_x,center_y)
            # result = self.If_clickMonster(frame_out,self.template)
            # print(result)


            cv2.imshow('frame', frame_out)           
            if cv2.waitKey(1) == ord('q'):
                break
    def Process_of_defeat(self):
        self.Keyboard_input("F1")
        time.sleep(1)
        self.Keyboard_input("space")
        return
    
    def Keyboard_input (self, keyboard):
        pyautogui.press(keyboard)
        return 
    def Mouse_movement(self,x,y):
        x,y = int(x),int(y)
        pydirectinput.moveTo(x, y)
        pydirectinput.click()
        return 
    def If_clickMonster(self,frame,template):  
        # frame = cv2.imread("./template/0001.jpg")
        # template = cv2.imread("./template/template1.jpg")
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(frame,None)
        kp2, des2 = sift.detectAndCompute(template,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        result = False
        threashold = 200
        if len(good) > 200 :
            result = True
        # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(frame,kp1,template,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
        # https://stackoverflow.com/questions/71473619/compute-similarity-measure-in-feature-matching-bfmatcher-in-opencv
        return result
    
    
script = Script()
script.Show_Screen()
# script.Process_of_defeat()

 
 