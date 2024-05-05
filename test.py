
def string_spliter(string):

    splite_number = 2
    string_list = []
    if splite_number ==3 :
        splitat = int(len(string)/3)
        left ,right = string[:splitat], string[splitat+splitat:]
        middle = string [splitat :-splitat]
        string_list.append(left)
        string_list.append(middle)
        string_list.append(right)
        return string_list
    if splite_number == 2 :
        splitat = int(len(string)/2)
        left = string[:splitat]
        right = string [splitat :]
        string_list.append(left)
        string_list.append(right)
        return string_list
    
print(string_spliter("Assassin Builder A"))