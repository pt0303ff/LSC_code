import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import torch

# === 檢查是否有可用的 GPU ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# === 1. 讀取資料 ===
csv_path = "1130moli_comments_parsed4_noL.csv"
df = pd.read_csv(csv_path)

# 與前面保持一致
TEXT_COL = "comment"
LABEL_COL = "roles"

df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

print("資料筆數：", len(df))
print(df[[TEXT_COL, LABEL_COL]].head())

# === 2. 載入 text2vec-large 模型 ===
# 這邊用常見的中文模型名稱，你也可以換成你自己的
MODEL_NAME = "GanymedeNil/text2vec-large-chinese"

model = SentenceTransformer(MODEL_NAME, device=device)   # 沒有 GPU 改成 "cpu"

# === 3. 產生 embedding ===
texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].astype(str).tolist()

print("開始產生 text2vec-large embedding ...")
# encode 一次會自動做 batch
X_t2v = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True   # 通常會做 L2 normalize
)

y = np.array(labels)

print("Embedding 形狀：", X_t2v.shape)

# === 4. 存成檔案 ===
np.save("emb_text2vec_large_X.npy", X_t2v)
# 標籤可以直接共用，也可再存一次
np.save("emb_labels.npy", y)

print("已儲存：emb_text2vec_large_X.npy, emb_labels.npy")

# === 5. Train/Test 切分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X_t2v, y, test_size=0.2, random_state=42, stratify=y
)

np.save("emb_text2vec_large_X_train.npy", X_train)
np.save("emb_text2vec_large_X_test.npy", X_test)
np.save("emb_text2vec_large_y_train.npy", y_train)
np.save("emb_text2vec_large_y_test.npy", y_test)

print("Train/Test 已分好並儲存。")
