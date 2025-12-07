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
**bge_m3_embedding.py** : 
**text2vec_embedding.py** : 

## **訓練SVM模型**
由於SVM適合高維度分析，embedding後維度有1024，因此選用此作為初步分類模型
**train_bge_svm.py**
**train_text2vec_svm.py**
SVM模型結果如下:

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
bge-m3 在 KOC（顧客）分類明顯優於 text2vec
顯示其在區分「純詢問購買型」語句上較具優勢。
text2vec-large 在 KOS（發話/分享者）表現明顯更好
表示 text2vec 對資訊性、開話題、評論式句型辨識更敏感。
KOF（粉絲）兩者表現都極高（~0.98–0.99）
粉絲型語句語氣明顯，兩模型都能穩定辨識。
整體 accuracy 完全一致：0.94
→ 兩模型皆適合用於直播留言角色分類。
若需要更均衡的多類別表現 → text2vec-large + SVM 更佳
若要最準確的 KOC 與 KOF 區分 → bge-m3 + SVM 更適合
