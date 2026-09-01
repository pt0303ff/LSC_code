# LSC_code
## **fb_comment.py** 
用來自動捲開影片留言區的留言，並且把所有留言抓出來

## **comment_parse.py**
用來把抓下來的留言依照"user"、"time"、"comment"分開來

## **1130moli_comments_parsed3.csv**
為目前先抓取FB:摩里沙卡木頭直播賣家11/30的直播存檔，且資料有按照"time"順序排好
[直播存檔連結](https://www.facebook.com/molisaka168168/videos/1580497086642651)

## **1130moli_comments_parsed4_noL.csv**
為1130moli_comments_parsed3.csv標記"roles"欄位後的資料，標記出了"KOC"、"KOS"、"KOF"、"other"，並且去除掉了主播(KOL)的留言

## **Embedding模型** 
``` bge_m3_embedding.py ```  
``` text2vec_embedding.py   ```  
```npy```  資料夾為embedding後向量檔案，同時也切割好訓練集、測試集供訓練分類模型使用  

## **訓練SVM模型**
由於SVM適合高維度分析，embedding後維度有1024，因此選用此作為初步分類模型  
``` train_bge_svm.py ```    
``` train_text2vec_svm.py ```  
   
**SVM模型結果如下:**       
  
| 類別                  | 指標              | **bge-m3 + SVM** | **text2vec-large + SVM** | 較佳模型               |
| ------------------- | --------------- | ---------------- | ------------------------ | ------------------ |
| **KOC（顧客）**         | Precision       | **1.0000**       | 0.9091                   | bge-m3             |
|                     | Recall          | **0.9688**       | 0.9375                   | bge-m3             |
|                     | F1-score        | **0.9841**       | 0.9231                   | **bge-m3（差距明顯）**   |
| **KOF（粉絲）**         | Precision       | 0.9890           | **0.9891**               | text2vec（兩者幾乎相同）   |
|                     | Recall          | 0.9783           | **0.9891**               | text2vec           |
|                     | F1-score        | 0.9836           | **0.9891**               | text2vec（差異極小）     |
| **KOS（開話題/分享者）**    | Precision       | 0.4000           | **0.6000**               | **text2vec 大幅優勝**  |
|                     | Recall          | 0.3333           | **0.5000**               | **text2vec**       |
|                     | F1-score        | 0.3636           | **0.5455**               | **text2vec（顯著提升）** |
| **Other（其他）**       | Precision       | **0.5000**       | 0.6000                   | text2vec           |
|                     | Recall          | **0.8000**       | 0.6000                   | bge-m3             |
|                     | F1-score        | 0.6154           | **0.6000（非常接近）**         | 幾乎相同               |
| **整體（All classes）** | Accuracy        | **0.9407**       | **0.9407**               | 兩者相同               |
|                     | Macro Avg F1    | 0.7367           | **0.7644**               | text2vec           |
|                     | Weighted Avg F1 | **0.9425**       | 0.9393                   | bge-m3（差異極小）       |


**⭐ 比較重點總結**
- bge-m3 在 KOC（顧客）分類明顯優於 text2vec  
  顯示其在區分「純詢問購買型」語句上較具優勢。  
 - text2vec-large 在 KOS（發話/分享者）表現明顯更好  
表示 text2vec 對資訊性、開話題、評論式句型辨識更敏感。  
- KOF（粉絲）兩者表現都極高（~0.98–0.99）  
粉絲型語句語氣明顯，兩模型都能穩定辨識。  
- 整體 accuracy 完全一致：0.94  
→ 兩模型皆適合用於直播留言角色分類。  
- 若需要更均衡的多類別表現 → text2vec-large + SVM 更佳  
- 若要最準確的 KOC 與 KOF 區分 → bge-m3 + SVM 更適合



## 26/02/20 簡單測試模型
```bge_test.py```   
測試bge模型實際分類效果用，輸入留言>輸出分類結果  
```text2vec_test.py```     
測試text2vec模型實際分類效果用，輸入留言>輸出分類結果  

## 動態角色分類分析
```dynamic_role.py```  
先調用embedding模型後，再利用訓練好的SVM模型預測，如果信心度<0.8，將調用LLM模型來辨識該則用戶留言之角色判定。  
整理同一用戶動態變化並產出:  
- ```transition.csv``` : 整理所有角色有變換的用戶變化過程  
- ```user_role_summary.csv``` : 整理該場直播所有用戶被統計的角色  
- ```role_count.csv``` : 統計KOC、KOF、KOS個別數量

## 加入0313直播資料
資料總共為1462筆留言，經過bge 、text2vec embedding與SVM訓練後結果為 :  
- bge-m3 + SVM  
 === Classification Report (LinearSVC, bge-m3) ===  
      
              precision    recall  f1-score   support

         KOC     0.9537    0.9717    0.9626       106
         KOF     0.9740    0.9434    0.9585       159
         KOS     0.5714    0.4444    0.5000         9
       other     0.5833    0.7368    0.6512        19

       accuracy                          0.9249       293
       macro avg     0.7706    0.7741    0.7681       293
       weighted avg  0.9290    0.9249    0.9260       293

   === Confusion Matrix ===    
      
          [[103   2   0   1]  
          [  2 150   2   5]  
          [  0   1   4   4]  
          [  3   1   1  14]]  

- text2vec + SVM  
 === Classification Report (LinearSVC, text2vec-large) ===  
      
               precision    recall  f1-score   support

         KOC     0.9130    0.9906    0.9502       106
         KOF     0.9867    0.9308    0.9579       159
         KOS     0.3333    0.1111    0.1667         9
       other     0.6000    0.7895    0.6818        19

       accuracy                         0.9181       293
       macro avg    0.7083    0.7055    0.6892       293
       weighted avg 0.9149    0.9181    0.9129       293
    
    === Confusion Matrix (SVM, text2vec-large) ===  

         [[105   0   1   0]
         [  5 148   1   5]
         [  3   0   1   5]
         [  2   2   0  15]]

## 26/05/07 更改SVM信心度計算方式、評估LLM修正準確率
```train_bge_svm.py```   
更新信心度計算方式，使用LinearSVC CalibratedClassifierCV  
```dynamic_role3.py```   
新的版本，更新SVM模型以及信心度調用LLM條件，低於閥值0.7  
```eva_LLM_intervantion.py```   
評估LLM介入修正的準確率，以及原本包含低信心度的準確率，及比較比率  

##26/08/20 更新後續驗證程式  
```0316bge_m3_calibrated_linear_svc_kocfkos.joblib```  
為最新SVM訓練模型檔案  
```dynamic_plot.py```  
分析並畫圖三個LLMs結果  
<img width="1080" height="500" alt="01_overall_svm_vs_hybrid_accuracy" src="https://github.com/user-attachments/assets/fd25a051-479a-4240-96c4-e82a22d80de1" />

```cali_trans.py```   
使用結果最佳模型qwen進一步以時間斷分析各角色轉變的結果，並劃出圖，目前0321直播分為六個時間斷分析
<img width="6257" height="3901" alt="overall_transition_users_calibrated" src="https://github.com/user-attachments/assets/13a93161-7359-43a0-858e-0c7fd3fc6183" />




