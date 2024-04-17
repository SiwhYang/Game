# License : CopyRight 
# 
# 
# 
# 

import cv2 as cv2
import numpy as np
from PIL import ImageGrab
import time
import pyautogui#, win32ui, win32con, win32api
from mss import mss
from PIL import Image
import threading 
# from cv_bridge import CvBridger

class Script():

    def __init__(self):
        return

    def Screen_Capture(self):
        bounding_box = {'top': 500, 'left': 300, 'width': 1000, 'height': 700}
        sct = mss()
        while (True):
            screenshot = np.array(sct.grab(bounding_box))
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            
            cv2.imshow('frame', screenshot)
            if cv2.waitKey(1) == ord('q'):
                break
            # a = cv2.imwrite("./data/000{}.jpg".format(i),screenshot)            
            # i = i + 1


    # def Screen_Capture(self):
    #     bounding_box = {'top': 500, 'left': 300, 'width': 1000, 'height': 700}
    #     sct = mss()
    #     image_list = []
    #     for i in range(0,10):
    #         screenshot = np.array(sct.grab(bounding_box))
    #         screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    #         image_list.append(screenshot)
    #     while(True):
    #         cv2.imshow('frame', image_list)
    #         if cv2.waitKey(1) == ord('q'):
    #             break
    #         self.Screen_Capture()
    #         self.Show_Screen()  

    #     return image_list
           


    def Screen_Analyzer(self):
        return

    def Show_Screen(self):
        
        while(True):
            image_list = self.Screen_Capture()
            cv2.imshow('frame', image_list)
            if cv2.waitKey(1) == ord('q'):
                break
            self.Screen_Capture()
            self.Show_Screen()  
        # Capture_thread = threading.Thread(target=script.Screen_Capture())
        # cap = cv2.VideoCapture("./data/%04d.jpg",cv2.CAP_IMAGES)
        # ret, frame = cap.read()
        # while (True):
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     cv2.imshow('frame', frame)
        #     if cv2.waitKey(1) == ord('q'):
        #         break
        #     self.Screen_Capture()
        #     self.Show_Screen()  
           

script = Script()
script.Screen_Capture()
# Capture_thread = threading.Thread(target=script.Screen_Capture())
# Show_thread = threading.Thread(target=script.Show_Screen())
# Capture_thread.start()
# Show_thread.start()
# Show_thread.join()


