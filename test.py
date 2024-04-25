import pydirectinput
import time

def string_spliter(string):
    string_list = []
    splitat = int(len(string)/3)
    left ,right = string[:splitat], string[splitat+splitat:]
    middle = string [splitat :-splitat]
    string_list.append(left)
    string_list.append(middle)
    string_list.append(right)
    return string_list
print(string_spliter("Palser Beta"))
# string_spliter("Assasin Builder A")
# string_spliter("RedDust")
# string_spliter("Palser Beta")