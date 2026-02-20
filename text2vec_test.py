import joblib
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ===== 1️⃣ 載入模型 =====
print("Loading SVM model...")
svm_model = joblib.load("text2vec_linear_svc_kocfkos.joblib")

print("Loading embedding model...")
embed_model = SentenceTransformer("GanymedeNil/text2vec-large-chinese")

# ===== 2️⃣ 類別名稱（依你當初 label 設定修改）=====


print("Model loaded successfully!\n")

# ===== 3️⃣ 互動測試 =====
while True:
    text = input("請輸入留言內容（輸入 exit 離開）：\n> ")

    if text.lower() == "exit":
        break

    # 產生 embedding
    embedding = embed_model.encode([text])

    # 預測
    prediction = svm_model.predict(embedding)[0]

    # decision score（可選）
    decision_score = svm_model.decision_function(embedding)

   

    print("\n🔍 預測角色：", prediction)
    print("📊 Decision score：", decision_score)
    print("-" * 50)