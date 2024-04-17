測試HWND
1. 測試在online上hwnd是否可以操作postmessenger or sendmessenegr
2. (不能用可能原因為有childhwnd)

if fail :
1. 改用pyautogui, 
2. 先關掉 imshow，確定roi能框到正確的東西
3. 再開始寫後面的操作


1. 處理記憶體升高爆炸問題
2. 圖像相似度比對