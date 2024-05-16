
Q: 
1. 大部分影像辨識都是使用videocapture, 但螢幕截圖無法產生該格式，使用workaround -> 存到硬碟再讀取，可以一次多張，但會很卡(IO太慢)，故只有一張。
2. 按鍵無法傳達到視窗，需要改用pydirect還有管理者模式-> 暫時用點擊螢幕，hwnd可以截圖但無法傳送點擊(當不在主螢幕時)。
3. matchingtemplate抓不到-> 因為尺寸不對，使用snipping抓圖解析度跑掉，改用paint直接截一樣的圖，或改用opencv feature-matching。
4. while loop使用 ~ 5min 記憶體會爆炸 -> mss instance 要手動關掉，garbage collection 不會自動處理 -> 用context manager (with 語法) 處理。
5. ROI太多，直接點該位置高機率點不到target，且會造成移動製造更多ROI，因此加入字串比對，用HSV將紅字濾出確定有確實指到target，再次點擊。
6. 用HSV濾出綠色確認ID位置，原意是當紅字與預設UI重疊時判斷不出，可點自身位置檢查是否點到target，但若將ID寫死可以當作權限鎖使用。

--------------------------------------------------
1. 將上面完整script做改裝，使其自動farming的同時自動輸出yolo標註的格式。
2. 已整合完成，將distor分為兩版，自動標記原版 與 yolo版 
3. 自動標記 -> 調整screen cutter，名字檢測，改寫 ,yolo->仍需關閉print

----------------------------------------------------
Before release, check
1. screen cutter 
2. if click monster parameter
