
import cv2 as cv2
import numpy as np
import time
from mss import mss
import pywintypes # // https://stackoverflow.com/questions/3956178/cant-load-pywin32-library-win32gui adding for import dll file to init win32gui
import pydirectinput
import pytesseract
from datetime import datetime
import sys
import warnings
import threading 
from pynput import keyboard
import win32api
import win32con
import os
import detect_api
import torch
from PIL import Image

warnings.filterwarnings("ignore")

Lincense_Text = "\nCopyright (c) 2024 Siwh of Meow Guild. All Right Reserved.\nThis software may not be copied, \
transmitted, provided or modified \n\
without permission of Siwh or members of Meow guild.\n"

Program_controller = True
Process_controller = True



class Script():
 
    def __init__(self,name,Monster_name,Monster_name_color, Use_Normal_attack ):
        self.Character_name = name
        self.Character_x_coordinate = None
        self.Character_y_coordinate = None
        self.Monster_name = Monster_name
        self.Monster_name_color = Monster_name_color
        self.Use_Normal_attack = Use_Normal_attack
        screen_x = None
        screen_y = None
        screen_roi_left = None
        screen_roi_width = None
        screen_roi_top = None
        screen_roi_height = None
        self.detector = None
        self.screen_spliter()
        self.template = cv2.imread("./template/template1.jpg")
        return
    
    def Screen_Capture_return(self):
        with mss() as sct :
            bounding_box = {'top': 1, 'left': 1, 'width': self.screen_x , 'height': self.screen_y}
            screenshot = sct.grab(bounding_box)
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
            arr = np.array(img)
            screenshot = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            mask = np.zeros(screenshot.shape,np.uint8)
            mask[self.screen_roi_top:self.screen_roi_height,self.screen_roi_left:self.screen_roi_width] = screenshot[self.screen_roi_top:self.screen_roi_height,self.screen_roi_left:self.screen_roi_width]
            
        return mask
    
    def Screen_Capture(self):    
        with mss() as sct :
            bounding_box = {'top': 1, 'left': 1, 'width': self.screen_x , 'height': self.screen_y}
            screenshot = np.array(sct.grab(bounding_box))
            mask = np.zeros(screenshot.shape,np.uint8)
            mask[self.screen_roi_top:self.screen_roi_height,self.screen_roi_left:self.screen_roi_width] = screenshot[self.screen_roi_top:self.screen_roi_height,self.screen_roi_left:self.screen_roi_width]
            cv2.imwrite("./data/0001.jpg",mask) 
        # https://stackoverflow.com/questions/54719730/when-taking-many-screenshots-with-mss-memory-fills-quickly-and-crashes-python
        # context manager
        return


    def Main(self):
        print(Lincense_Text)
        count = 0
        global Program_controller
        global Process_controller
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        result,self.Character_x_coordinate, self.Character_y_coordinate = self.Refresh_and_Process_myself_screen()
        if result == False :
            Program_controller = False
            print("Character name error, please check character name is correct and visible !")
            print("Press 'p' to stop")
            return
        print("Character name = '{}' confirmed".format(self.Character_name) )
        print("Monster name = '{}' confirmed".format( self.Monster_name) )
        print("Initializing...")
        self.detector = detect_api.detectapi(weights="./weights/Ore_0.03.pt",img_size=416)
        Moster_selected = False
        while(Program_controller):
            if (Process_controller) :
                # print(Moster_selected)
                frame_out,x_list,y_list,x_size,y_size = self.Refresh_and_Process_screen(object_detector)  
                if Moster_selected == False :
                    self.Keyboard_input('F2')
                    self.Keyboard_input('Esc') 
                    # _,Character_x_position,Character_y_position = self.Refresh_and_Process_myself_screen() # check myself
                    self.Mouse_Click( self.Character_x_coordinate,self.Character_y_coordinate) # check myself
                    frame_out,_,_,_,_= self.Refresh_and_Process_screen(object_detector)  # check myself
                    check_if_click = self.If_clickMonster(frame_out,self.template) # check myself
                    if check_if_click == True : # check myself
                        Moster_selected = True  
                        continue   # check myself
                    print("Try to find {}".format(self.Monster_name))
                    if len(x_list) < 30 : 
                        for i in range(0,len(x_list)): # Try to click monster
                            self.Mouse_movement(x_list[i],y_list[i])
                            time.sleep(0.1)
                            _,Findtext = self.Refresh_and_Process_Name_screen()
                            if Findtext == True:
                                # self.Yolo_Labelling (frame_out,x_list[i],y_list[i],x_size[i],y_size[i])
                                self.Mouse_Click(x_list[i],y_list[i])
                                self.Mouse_Click(x_list[i],y_list[i])
                                frame_out,_,_,_,_= self.Refresh_and_Process_screen(object_detector)  
                                check_if_click = self.If_clickMonster(frame_out,self.template)
                                if check_if_click == True :
                                    print("We found {} ".format(self.Monster_name))
                                    Moster_selected = True  
                                    break             
                                else :  
                                    Moster_selected == False
                            else :  
                                Moster_selected == False
                    else : 
                        print("Wait for the screen to stabilize")

                        # cv2.imshow('frame', frame_out)  
                        # if cv2.waitKey(1) == ord('q'):
                        #     break         
                elif Moster_selected == True: 
                    print("Start killing {} ".format(self.Monster_name))
                    check_if_click = self.If_clickMonster(frame_out,self.template)
                    while (check_if_click) : 
                        frame_out,_,_,_,_= self.Refresh_and_Process_screen(object_detector)   
                        self.Defeating_Process()
                        check_if_click = self.If_clickMonster(frame_out,self.template)
                        # cv2.imshow('frame', frame_out)           
                    # self.Keyboard_input('F2')
                    print("Defeated {}, looting ".format(self.Monster_name))
                    self.Keyboard_press('space',3)
                    count = count + 1
                    print("We have killed {} {}  ".format(count,self.Monster_name))
                    Moster_selected = False
                    # cv2.imshow('f rame', frame_out)
                    # if cv2.waitKey(1) == ord('q'):
                    #         break                    

    def string_spliter(self,string):
        string_list = []
        splitat = int(len(string)/3)
        left ,right = string[:splitat], string[splitat+splitat:]
        middle = string [splitat :-splitat]
        string_list.append(left)
        string_list.append(middle)
        string_list.append(right)
        return string_list


    def screen_spliter(self):
        screen_width = win32api.GetSystemMetrics(0)
        screen_length = win32api.GetSystemMetrics(1)
        screen_roi_x = screen_width 
        screen_roi_y = screen_length
        self.screen_x = screen_roi_x
        self.screen_y = screen_roi_y
        # cut_ratio = 0.24
        cut_ratio = 0
        self.screen_roi_left = int(screen_roi_x * cut_ratio)
        self.screen_roi_width = screen_width
        self.screen_roi_top = 1
        self.screen_roi_height = screen_roi_y - 30

    def Yolo_Labelling (self,frame,C_x,C_y,Size_x,Size_y):
        name = self.Monster_name
        if_findtarget = False
        count_lines = 0
        with open("./Yolo/classes.txt","r") as f:
            for line in f.readlines():
                if name in line.rstrip() :
                    count_lines = count_lines + 1
                    if_findtarget = True
                    break 
                else :
                    count_lines = count_lines + 1
        if (if_findtarget == False) :
            with open("./Yolo/classes.txt","a") as f :
                f.writelines(name +'\n')
                count_lines = count_lines + 1
        
        if not os.path.exists("./Yolo/{}".format(self.Monster_name)):
             os.makedirs("./Yolo/{}".format(self.Monster_name))
        
        # timestamp
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        # print("date and time =", dt_string)
        
        # // output 1. image  
        frame_roi = frame.copy()
        cv2.imwrite("./Yolo/{}/{}.jpg".format(self.Monster_name,dt_string),frame)
        # //2. image with roi drawing (checking)
        x = int((2*C_x-Size_x)/2)
        y = int((2*C_y-Size_y)/2)
        w = Size_x
        h = Size_y
        cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (0, 0, 200), 3)
        cv2.imwrite("./Yolo/{}/{}-roi.jpg".format(self.Monster_name,dt_string),frame_roi)
        # // 3. yolo.txt format (roi coordinate)
        C_x_ratio = C_x/self.screen_x
        C_y_ratio = C_y/self.screen_y
        Size_x_ratio = Size_x/self.screen_x
        Size_y_ratio = Size_y/self.screen_y
        write_line = "{} {} {} {} {}".format(count_lines,C_x_ratio,C_y_ratio,Size_x_ratio,Size_y_ratio)
        with open("./Yolo/{}/{}.txt".format(self.Monster_name,dt_string),"a") as f :
            f.writelines(write_line +'\n')
        
        return 




    def Refresh_and_Process_myself_screen(self):
        print("Start checking character's name")
        # self.Screen_Capture()
        # cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        # ret, frame = cap.read()
        frame = self.Screen_Capture_return()
        frame_out = frame.copy()
        hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
        def fixHSVRange(h, s, v):
            # Normal H,S,V: (0-360,0-100%,0-100%)
            # OpenCV H,S,V: (0-180,0-255 ,0-255)
            return (180 * h / 360, 255 * s / 100, 255 * v / 100)
        
       
        lower = np.array([fixHSVRange(140,10,0)])
        upper = np.array([fixHSVRange(160,100,100)])
        text_hsv = cv2.inRange(hsv,lower, upper )

        # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_data(text_hsv,lang = 'eng',output_type='dict')
        boxes = len(text['level'])
        Result = False
        center_x_click = None
        center_y_click = None
        for i in range(boxes):
            if text['text'][i] != '':      
                Target_string = self.string_spliter(self.Character_name)
                for j in range(0,len(Target_string)) : 
                    if Target_string[j] in text['text'][i] :
                        Result = True
                        center_x_click = int(text['left'][i] + text['width'][i]/2)
                        center_y_click = int(text['top'][i] + text['height'][i]/2)
        if center_x_click != None and  center_y_click != None :
            return Result, center_x_click, center_y_click+10 #, text_hsv
        else : return Result, 0 ,0 

    def Refresh_and_Process_Name_screen(self):
        # self.Screen_Capture()
        # cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        # ret, frame = cap.read()
        frame = self.Screen_Capture_return()
        frame_out = frame.copy()
        
        hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
        if self.Monster_name_color == "Red" :
            lower = np.array([155,25,100])
            upper = np.array([343,255,255])
            text_hsv = cv2.inRange(hsv,lower, upper )
        
        elif self.Monster_name_color != "Red" :
            lower = np.array([(0,0,70)])
            upper = np.array([(180,30,200)])
            text_hsv = cv2.inRange(hsv,lower, upper )

        pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_string(text_hsv,lang = 'eng')
        Result = False  
        Target_string = self.string_spliter(self.Monster_name)
        for j in range(0,len(Target_string)) : 
            if Target_string[j] in text :
                Result = True

        return text_hsv,Result

    def Refresh_and_Process_screen(self,object_detector):
        frame = self.Screen_Capture_return()
        with torch.no_grad() :
            result,names = self.detector.detect([frame])
            img = result[0][0]
            center_x_click = []
            center_y_click = []
            for cls,(x1,y1,x2,y2),conf in result[0][1]:
                print(names[cls],x1,y1,x2,y2,conf)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
                center_x = (x1+x2)/2
                center_y = (y1+y2)/2
                center_x_click.append(center_x)
                center_y_click.append(center_y)
        return img, center_x_click, center_y_click, 0 , 0

    

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
        threashold = 70
        if len(good) > threashold :
            result = True
        # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(frame,kp1,template,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
        # https://stackoverflow.com/questions/71473619/compute-similarity-measure-in-feature-matching-bfmatcher-in-opencv
        return result

            

    def Defeating_Process(self):
        if self.Use_Normal_attack == "1" :
            # print(self.Use_Normal_attack)
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
        count_time = 0
        intervel = 0.5
        while (count_time < second):
            pydirectinput.press(keyboard)
            time.sleep(intervel)
            pydirectinput.keyUp(keyboard)
            count_time = count_time + intervel

    def Mouse_Click(self,x,y):
        if x == None and y == None:
            pass
        else :
            x,y = int(x),int(y)
            pydirectinput.moveTo(x, y)
            pydirectinput.click()
        return 

    def Mouse_movement(self,x,y):
        x,y = int(x),int(y)
        pydirectinput.moveTo(x, y)
        # pydirectinput.click()
        return 

    def test_screen(self):
        
        a = detect_api.detectapi(weights="./weights/ABA_0.15.pt",img_size=416)
        with torch.no_grad():
            while(True):
                # self.Screen_Capture()    
                # cap = cv2.imread("./data/0001.jpg")
                cap = self.Screen_Capture_return()
                result,names = a.detect([cap])
                img = result[0][0]
                for cls,(x1,y1,x2,y2),conf in result[0][1]:
                    print(names[cls],x1,y1,x2,y2,conf)
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
                    cv2.putText(img,names[cls],(x1,y1-30),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),thickness=2)
                cv2.imshow('video',img)
                if cv2.waitKey(1) == ord('q'):
                    break   
        return img 
    

    def Show_Screen(self):
        while(True):
            frame = self.test_screen()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                    break   

class Keystrokes_Monitor () :
    def on_release(self,key):
        global Program_controller
        global Process_controller
        if Process_controller == True :
            if key == keyboard.KeyCode.from_char('p') :
                print("Pause ! press 'o' to cloase program, press 's' to restart. ")
                Process_controller = False
        if Process_controller == False :
            if key == keyboard.KeyCode.from_char('s') :
                print("Restart ! press 'p' to pause. ")
                Process_controller = True
            if key == keyboard.KeyCode.from_char('o') :
                # Stop listener
                Program_controller = False
                return False
            

if __name__ == '__main__':
   
    # script = Script("RedDust","Assassin Builder A", "Gray", 1) # Assassin_Builder_A
    # script.test_screen()
    # # script.Yolo_Labelling(1,1,1,1,1)

   
    name_split = sys.argv[2].split("_")
    monster_name = ' '.join (name_split)
    Main_Process = Script(sys.argv[1],monster_name,sys.argv[3],sys.argv[4])
    # Main_Process = Script("RedDust","Assassin Builder A", "Red", 1) # Assassin_Builder_A
    Keystrokes_Monitor = Keystrokes_Monitor()
    Main_Process_thread = threading.Thread( target=Main_Process.Main)
# Collect events until released
    with keyboard.Listener(on_release=Keystrokes_Monitor.on_release) as Keystrokes_Monitor_thread:
        Main_Process_thread.start()
        Keystrokes_Monitor_thread.join()