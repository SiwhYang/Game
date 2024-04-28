
import cv2 as cv2
import numpy as np
import time
from mss import mss
import pywintypes # // https://stackoverflow.com/questions/3956178/cant-load-pywin32-library-win32gui adding for import dll file to init win32gui
import pydirectinput
import pytesseract
import sys
import warnings
warnings.filterwarnings("ignore")

class Script():
 
    def __init__(self,name,Monster_name):
        self.Character_name = name
        self.Character_x_coordinate = None
        self.Character_y_coordinate = None
        self.Monster_name = Monster_name
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
        count = 0
        print("Initializing...")
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        result,self.Character_x_coordinate, self.Character_y_coordinate = self.Refresh_and_Process_myself_screen()
        if result == False :
            print("Character name error, please check character name is correct and visible !")
            return
        print("Character name = '{}' confirmed".format(self.Character_name) )
        print("Monster name = '{}' confirmed".format( self.Monster_name) )
        Moster_selected = False
        while(True):
            # print(Moster_selected)
            frame_out,x_list,y_list = self.Refresh_and_Process_screen(object_detector)  
            if Moster_selected == False :
                self.Keyboard_input('F2')
                self.Keyboard_input('Esc') 
                # _,Character_x_position,Character_y_position = self.Refresh_and_Process_myself_screen() # check myself
                self.Mouse_Click( self.Character_x_coordinate,self.Character_y_coordinate) # check myself
                frame_out,_,_= self.Refresh_and_Process_screen(object_detector)  # check myself
                check_if_click = self.If_clickMonster(frame_out,self.template) # check myself
                if check_if_click == True : # check myself
                    Moster_selected = True  
                    continue   # check myself
                print("Try to find {}".format(self.Monster_name))
                for i in range(0,len(x_list)): # Try to click monster
                    self.Mouse_movement(x_list[i],y_list[i])
                    time.sleep(0.1)
                    _,Findtext = self.Refresh_and_Process_Name_screen()
                    if Findtext == True:
                        self.Mouse_Click(x_list[i],y_list[i])
                        self.Mouse_Click(x_list[i],y_list[i])
                        frame_out,_,_= self.Refresh_and_Process_screen(object_detector)  
                        check_if_click = self.If_clickMonster(frame_out,self.template)
                        
                        if check_if_click == True :
                            print("We found {} ".format(self.Monster_name))
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
                print("Start killing {} ".format(self.Monster_name))
                check_if_click = self.If_clickMonster(frame_out,self.template)
                while (check_if_click) : 
                    frame_out,_,_= self.Refresh_and_Process_screen(object_detector)   
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

    def Refresh_and_Process_myself_screen(self):
        print("Start checking character's name")
        self.Screen_Capture()
        cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        ret, frame = cap.read()
        frame_out = frame.copy()
        hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
        def fixHSVRange(h, s, v):
            # Normal H,S,V: (0-360,0-100%,0-100%)
            # OpenCV H,S,V: (0-180,0-255 ,0-255)
            return (180 * h / 360, 255 * s / 100, 255 * v / 100)
        
        # lower = np.array([fixHSVRange(150,40,50)])
        # upper = np.array([fixHSVRange(170,100,100)])
        lower = np.array([fixHSVRange(150,10,50)])
        upper = np.array([fixHSVRange(170,180,100)])
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
        self.Screen_Capture()
        cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        ret, frame = cap.read()
        frame_out = frame.copy()
        
        hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
        lower = np.array([155,25,100])
        upper = np.array([343,255,255])
        text_hsv = cv2.inRange(hsv,lower, upper )
        # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = '.\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_string(text_hsv,lang = 'eng')
        
        # Result = False
        # print(text)
        # if 'sa' in text or 'Bui' in text or 'Ass' in text or 'der' in text or 'Cal' in text:
        # if 'Cali' in text or 'Alpha' in text or 'Pl' in text or 'Hobo' in text:
        # if 'Crystal' in text or 'Ore' in text  :
            # Result = True
        Result = False  
        Target_string = self.string_spliter(self.Monster_name)
        for j in range(0,len(Target_string)) : 
            if Target_string[j] in text :
                Result = True

        return text_hsv,Result

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
        min_contour_area = 300  # Define your minimum area threshold > 500 ABA > 300 Ore
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        center_x_click = []
        center_y_click = []
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = (x*2 + w)/2
            center_y = (y*2 + h)/2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            center_x_click.append(center_x)
            center_y_click.append(center_y)

        return frame, center_x_click, center_y_click, 
    

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
        threashold = 50
        if len(good) > threashold :
            result = True
        # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(frame,kp1,template,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
        # https://stackoverflow.com/questions/71473619/compute-similarity-measure-in-feature-matching-bfmatcher-in-opencv
        return result

            
    def Defeating_Process(self):
        # self.Keyboard_input("space")
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

    



    def test_screen(self,object_detector):
        self.Screen_Capture()
        cap = cv2.VideoCapture("./data/0001.jpg",cv2.CAP_IMAGES)
        ret, frame = cap.read()
        frame_out = frame.copy()
        hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
        def fixHSVRange(h, s, v):
            # Normal H,S,V: (0-360,0-100%,0-100%)
            # OpenCV H,S,V: (0-180,0-255 ,0-255)
            return (180 * h / 360, 255 * s / 100, 255 * v / 100)
        
        lower = np.array([fixHSVRange(150,10,50)])
        upper = np.array([fixHSVRange(170,180,100)])
        text_hsv = cv2.inRange(hsv,lower, upper )
        return text_hsv #, center_x_click, center_y_click, 
    

    def Show_Screen(self):
        object_detector = cv2.createBackgroundSubtractorMOG2() 
        while(True):
            frame = self.test_screen(object_detector)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                    break   



if __name__ == '__main__':
    name_split = sys.argv[2].split("_")
    monster_name = ' '.join (name_split)
    script = Script(sys.argv[1],monster_name)
    script.Main()

    # script = Script("RedDust","Assassin Builder A") # Assassin_Builder_A
    # script.Show_Screen()


