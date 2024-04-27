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

class Auto_Run():
    def __init__(self,name,Monster_name):
        # self.Character_name = name
        # self.Monster_name = Monster_name
        # self.template = cv2.imread("./template/template1.jpg")
        return
    def Screen_Capture(self):
        with mss() as sct :
            bounding_box = {'top': 1, 'left': 1, 'width': 1300, 'height': 1000}
            screenshot = np.array(sct.grab(bounding_box))
            cv2.imwrite("./data/0001.jpg",screenshot) 
        # https://stackoverflow.com/questions/54719730/when-taking-many-screenshots-with-mss-memory-fills-quickly-and-crashes-python
        # context manager
        return
    
    def test_screen(self,object_detector):
        # self.Screen_Capture()
        # cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        # ret, frame = cap.read()
        # frame_out = frame.copy()
        # gray = object_detector.apply(frame) 
        # _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # dilated = cv2.dilate(mask_eroded,cv2. getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)

        # contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        # min_contour_area = 1000  # Define your minimum area threshold
        # large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_contour_area]
        
        # center_x_click = []
        # center_y_click = []
        # for cnt in large_contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     center_x = (x*2 + w)/2
        #     center_y = (y*2 + h)/2
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
        #     center_x_click.append(center_x)
        #     center_y_click.append(center_y)

        self.Screen_Capture()
        cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        ret, frame = cap.read()
        frame_out = frame.copy()
        hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
        def fixHSVRange(h, s, v):
            # Normal H,S,V: (0-360,0-100%,0-100%)
            # OpenCV H,S,V: (0-180,0-255 ,0-255)
            return (180 * h / 360, 255 * s / 100, 255 * v / 100)
        
        lower = np.array([fixHSVRange(150,40,50)])
        upper = np.array([fixHSVRange(170,100,100)])
        text_hsv = cv2.inRange(hsv,lower, upper )

        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        # text = pytesseract.image_to_data(text_hsv,lang = 'eng',output_type='dict')
        # boxes = len(text['level'])
        # Result = False
        # for i in range(boxes):
        #     if text['text'][i] != '':
        #         # print(text['left'][i], text['top'][i], text['width'][i], text['height'][i], text['text'][i])        
        #         Target_string = self.string_spliter(self.Character_name)
        #         for j in range(0,len(Target_string)) : 
        #             if Target_string[j] in text['text'][i] :
        #                 Result = True
        #                 center_x_click = int(text['left'][i] + text['width'][i]/2)
        #                 center_y_click = int(text['top'][i] + text['height'][i]/2)
        #                 # self.Mouse_movement(center_x_click,center_y_click)
        return frame #, center_x_click, center_y_click, 
    

    def Show_Screen(self):
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        while(True):
            frame = self.test_screen(object_detector)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                    break   
Auto_Run = Auto_Run("yy","s")
Auto_Run.Show_Screen()