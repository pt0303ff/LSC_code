import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import joblib

# === 1. 載入 text2vec 的 train / test 資料 ===
X_train = np.load("npy/emb_text2vec_large_X_train.npy")
X_test = np.load("npy/emb_text2vec_large_X_test.npy")

# 標籤如果是字串，要加 allow_pickle=True
y_train = np.load("npy/emb_text2vec_large_y_train.npy", allow_pickle=True)
y_test = np.load("npy/emb_text2vec_large_y_test.npy", allow_pickle=True)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train 分布:", Counter(y_train))
print("y_test 分布:", Counter(y_test))

# === 2. 建立 Linear SVM 分類器 ===
clf = LinearSVC(
    class_weight="balanced",  # 類別不均衡時加權
    random_state=42
)

# === 3. 訓練 ===
print("開始訓練 LinearSVC（text2vec-large）...")
clf.fit(X_train, y_train)

# === 4. 在測試集上評估 ===
y_pred = clf.predict(X_test)

print("\n=== Classification Report (LinearSVC, text2vec-large) ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== Confusion Matrix (SVM, text2vec-large) ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# === 5. 儲存模型與預測結果（之後可視覺化用） ===
joblib.dump(clf, "text2vec_linear_svc_kocfkos.joblib")
print("\n模型已儲存為 text2vec_linear_svc_kocfkos.joblib")

np.save("npy/text2vec_svm_y_test.npy", y_test)
np.save("npy/text2vec_svm_y_pred.npy", y_pred)
np.save("npy/text2vec_svm_cm.npy", cm)
print("已儲存 text2vec SVM 的預測結果與混淆矩陣。")
