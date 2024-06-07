 
# import pydirectinput

    
# def Keyboard_input (keyboard):
#     pydirectinput.press(keyboard)
#     pydirectinput.keyUp(keyboard)
#     return 

# while (True):
#     Keyboard_input("F1")
#     Keyboard_input("space")  

  
import requests

url = 'https://notify-api.line.me/api/notify'
token = 'L89JILLN8ly8O3ctgiaKh0rSKDtOBhjthIW4f5z1qnx'
headers = {
    'Authorization': 'Bearer ' + token    # 設定權杖
}

def line_Notify(Messenger):
    data = {
        'message': str(Messenger)     # 設定要發送的訊息
    }
    data = requests.post(url, headers=headers, data=data)   # 使用 POST 方法

