 
import pydirectinput

    
def Keyboard_input (keyboard):
    pydirectinput.press(keyboard)
    pydirectinput.keyUp(keyboard)
    return 

while (True):
    Keyboard_input("F1")
    # Keyboard_input("space")  
