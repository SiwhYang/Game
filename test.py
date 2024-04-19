import pydirectinput
import time

while (True):
    pydirectinput.press('F1')
    time.sleep(1)
    pydirectinput.keyUp('F1')