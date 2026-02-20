import joblib
import numpy as np
from FlagEmbedding import BGEM3FlagModel


# ===== 1️⃣ 載入模型 =====
print("Loading SVM model...")
svm_model = joblib.load("bge_m3_linear_svc_kocfkos.joblib")

print("Loading BGE-M3 embedding model...")
embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')


print("Model loaded successfully!\n")

# ===== 3️⃣ 互動測試 =====
while True:
    text = input("請輸入留言內容（輸入 exit 離開）：\n> ")

    if text.lower() == "exit":
        break

    embedding = embed_model.encode([text])["dense_vecs"]

    prediction = svm_model.predict(embedding)[0]
    decision_score = svm_model.decision_function(embedding)

    print("\n🔍 預測角色：", prediction)
    print("📊 Decision score：", decision_score)
    print("-" * 50)