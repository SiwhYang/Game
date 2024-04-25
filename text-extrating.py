
import cv2 as cv2
import numpy as np
from PIL import ImageGrab
import time
from mss import mss
import pywintypes # // https://stackoverflow.com/questions/3956178/cant-load-pywin32-library-win32gui adding for import dll file to init win32gui
import win32gui
import pydirectinput
import matplotlib.pyplot as plt
import pytesseract


class Script():
 
    def __init__(self):
        self.template = cv2.imread("./template/template1.jpg")
        return
  
    def Screen_Capture(self):
        with mss() as sct :
            bounding_box = {'top': 1, 'left': 1, 'width': 1300, 'height': 1000}
            screenshot = np.array(sct.grab(bounding_box))
            cv2.imwrite("./data/0001.jpg",screenshot) 
        # https://stackoverflow.com/questions/54719730/when-taking-many-screenshots-with-mss-memory-fills-quickly-and-crashes-python
        # context manager
        return


    def Main(self):
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        Moster_selected = False
        while(True):
            # print(Moster_selected)
            frame_out,x_list,y_list = self.Refresh_and_Process_screen(object_detector)  
            if Moster_selected == False :
                self.Keyboard_input('F2')
                self.Keyboard_input('Esc')
                for i in range(0,len(x_list)): 
                    self.Mouse_movement(x_list[i],y_list[i])
                    time.sleep(0.1)
                    _,Findtext = self.Refresh_and_Process_Name_screen()
                    if Findtext == True:
                        self.Mouse_Click(x_list[i],y_list[i])
                        self.Mouse_Click(x_list[i],y_list[i])
                        frame_out,_,_= self.Refresh_and_Process_screen(object_detector)  
                        check_if_click = self.If_clickMonster(frame_out,self.template)
                        
                        if check_if_click == True :
                            print("we ready to change status")
                            Moster_selected = True  
                            break             
                        else :  
                            Moster_selected == False
                    else :  
                        Moster_selected == False

                    # cv2.imshow('frame', frame_out)  
                    # if cv2.waitKey(1) == ord('q'):
                    #     break         
            elif Moster_selected == True: 
                check_if_click = self.If_clickMonster(frame_out,self.template)
                while (check_if_click) : 
                    frame_out,_,_= self.Refresh_and_Process_screen(object_detector)   
                    self.Defeating_Process()
                    check_if_click = self.If_clickMonster(frame_out,self.template)
                    # cv2.imshow('frame', frame_out)           
                # self.Keyboard_input('F2')
                self.Keyboard_press('space',3)
                Moster_selected = False
                # cv2.imshow('f rame', frame_out)
                # if cv2.waitKey(1) == ord('q'):
                #         break                    

    def Refresh_and_Process_Name_screen(self):
        self.Screen_Capture()
        cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        ret, frame = cap.read()
        frame_out = frame.copy()
        
        # _,text = cv2.threshold(frame_out,100,255,cv2.THRESH_BINARY)
        # hsv = cv2.cvtColor(text, cv2.COLOR_BGR2HSV)
        # text_hsv = cv2.inRange(hsv,(0, 254, 254), (1, 255, 255) )
        # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        # text = pytesseract.image_to_string(text_hsv,lang = 'eng')
        hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
        lower = np.array([155,25,100])
        upper = np.array([343,255,255])
        text_hsv = cv2.inRange(hsv,lower, upper )
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_string(text_hsv,lang = 'eng')
        
        Result = False
        print(text)
        if 'sa' in text or 'Bui' in text or 'Ass' in text or 'der' in text or 'Cal' in text:
        # if 'Cali' in text or 'Alpha' in text or 'Pl' in text or 'Hobo' in text:
        # if 'Crystal' in text or 'Ore' in text  :
            Result = True

        return text_hsv,Result
    

    def test_screen(self,object_detector):
        self.Screen_Capture()
        cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        ret, frame = cap.read()
        frame_out = frame.copy()
        gray = object_detector.apply(frame) 
        _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(mask_eroded,cv2. getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)

        contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        min_contour_area = 1000  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_contour_area]
        
        center_x_click = []
        center_y_click = []
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = (x*2 + w)/2
            center_y = (y*2 + h)/2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            center_x_click.append(center_x)
            center_y_click.append(center_y)

        return frame #, center_x_click, center_y_click, 
    


    def Refresh_and_Process_screen(self,object_detector):
        self.Screen_Capture()
        cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        ret, frame = cap.read()
        frame_out = frame.copy()
        gray = object_detector.apply(frame) 
        _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(mask_eroded,cv2. getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)

        contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        min_contour_area = 500  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        # large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_contour_area]
        
        center_x_click = []
        center_y_click = []
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = (x*2 + w)/2
            center_y = (y*2 + h)/2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            center_x_click.append(center_x)
            center_y_click.append(center_y)
        

        # _,text = cv2.threshold(frame_out,100,255,cv2.THRESH_BINARY)
        # hsv = cv2.cvtColor(text, cv2.COLOR_BGR2HSV)
        # text_hsv = cv2.inRange(hsv,(0, 254, 254), (1, 255, 255) )
        # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        # text = pytesseract.image_to_string(text_hsv,lang = 'eng')
        
        # Result = False
        # print(text)
        # if 'sa' in text or 'Bui' in text or 'Ass' in text or 'der' in text:
        #     Result = True

        return frame, center_x_click, center_y_click, 
    
    def Show_Screen(self):
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        while(True):
            frame = self.test_screen(object_detector)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                    break   
            
    def Defeating_Process(self):
        self.Keyboard_input("space")
        self.Keyboard_input("F1")
        time.sleep(0.1) 
        # self.Keyboard_input("Esc")
        return
    
    def Keyboard_input (self, keyboard):
        pydirectinput.press(keyboard)
        time.sleep(1)
        pydirectinput.keyUp(keyboard)
        return 
    
    def Keyboard_press (self, keyboard, second):
        interval = 0.1
        second = 1
        loop_count = int(second/interval)
        for i in range (0,loop_count):
            pydirectinput. press(keyboard)
            time.sleep(interval)
            pydirectinput.keyUp(keyboard)
            i = i + 1

    def Mouse_Click(self,x,y):
        x,y = int(x),int(y)
        pydirectinput.moveTo(x, y)
        pydirectinput.click()
        return 

    def Mouse_movement(self,x,y):
        x,y = int(x),int(y)
        pydirectinput.moveTo(x, y)
        # pydirectinput.click()
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
        # print(len(good))
        threashold = 150
        if len(good) > threashold :
            result = True
        # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(frame,kp1,template,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
        # https://stackoverflow.com/questions/71473619/compute-similarity-measure-in-feature-matching-bfmatcher-in-opencv
        return result
    
    
script = Script()
# script.Show_Screen() 
script.Main()
