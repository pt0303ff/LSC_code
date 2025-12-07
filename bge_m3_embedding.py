import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.model_selection import train_test_split
import torch
from collections import Counter

# === 檢查是否有可用的 GPU ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# === 1. 讀取資料 ===
csv_path = "1130moli_comments_parsed4_noL.csv"
df = pd.read_csv(csv_path)

# 改成你的欄位名稱
TEXT_COL = "comment"   # 留言文字欄位
LABEL_COL = "roles"    # 標籤欄位（KOC/KOF/KOS）

# 過濾掉缺文字或缺標籤的列
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

print("資料筆數：", len(df))
print(df[[TEXT_COL, LABEL_COL]].head())

# === 2. 載入 bge-m3 模型 ===
# 常用 repo 名稱："BAAI/bge-m3"
model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=True,           # 如果 GPU 記憶體不夠可改成 False
    device=device            # 如果沒有 GPU 就改成 'cpu'
)

# === 3. 定義一個小工具，把文字轉 embedding ===
def get_embeddings_bgem3(text_list, batch_size=64):
    """
    輸入：list of str
    輸出：numpy array, shape = (N, embedding_dim)
    """
    all_vecs = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        # encode 返回 dict，dense_vecs 是句向量
        out = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        vecs = out["dense_vecs"]  # numpy array (batch, dim)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)

# === 4. 對整份資料做 embedding ===

texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].astype(str).tolist()

print("開始產生 bge-m3 embedding ...")
X_bge = get_embeddings_bgem3(texts, batch_size=64)
y = np.array(labels)

print("Embedding 形狀：", X_bge.shape)  # (樣本數, 維度)

# === 5. 存成檔案，之後可以重複使用 ===
np.save("emb_bge_m3_X.npy", X_bge)
np.save("emb_labels.npy", y)   # 這個可以共用給兩個模型

print("已儲存：emb_bge_m3_X.npy, emb_labels.npy")
print("標籤分布：")
print(Counter(y))

# === 6. 順便切 train/test，之後可以直接拿來訓練分類器 ===
X_train, X_test, y_train, y_test = train_test_split(
    X_bge, y, test_size=0.2, random_state=42, stratify=y
)

np.save("emb_bge_m3_X_train.npy", X_train)
np.save("emb_bge_m3_X_test.npy", X_test)
np.save("emb_bge_m3_y_train.npy", y_train)
np.save("emb_bge_m3_y_test.npy", y_test)

print("Train/Test 已分好並儲存。")
