
Q: 
1. 大部分影像辨識都是使用videocapture, 但螢幕截圖無法產生該格式，使用workaround -> 存到硬碟再讀取，可以一次多張，但會很卡(IO太慢)，故只有一張
2. 按鍵無法傳達到視窗，需要改用pydirect還有管理者模式-> 暫時用點擊螢幕，hwnd可以截圖但無法傳送點擊(當不在主螢幕時)
3. matchingtemplate抓不到-> 因為尺寸不對，使用snipping抓圖解析度跑掉，改用paint直接截一樣的圖，或改用opencv feature-matching
4. while loop使用 ~ 5min 記憶體會爆炸 -> mss instance 要手動關掉，garbage collection 不會自動處理 -> 用context manager (with 語法) 處理
5. ROI太多，改用字串比對
6. 優化code結構，把refresh，main_process,defeating process，等等拆開來寫，以便更好的執行後續邏輯