import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import joblib
from sklearn.linear_model import LogisticRegression

# === 1. 載入 train / test 資料 ===
X_train = np.load("npy/emb_bge_m3_X_train.npy")
X_test = np.load("npy/emb_bge_m3_X_test.npy")

# label 如果是字串，記得加 allow_pickle=True
y_train = np.load("npy/emb_bge_m3_y_train.npy", allow_pickle=True)
y_test = np.load("npy/emb_bge_m3_y_test.npy", allow_pickle=True)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train 分布:", Counter(y_train))
print("y_test 分布:", Counter(y_test))

# === 2. 建立 Linear SVM 分類器 ===
clf = LinearSVC(
    class_weight="balanced",  # 類別不均時很有用
    random_state=42
)

# === 3. 訓練 ===
print("開始訓練 LinearSVC...")
clf.fit(X_train, y_train)

# === 4. 在測試集上評估 ===
y_pred = clf.predict(X_test)

print("\n=== Classification Report (LinearSVC, bge-m3) ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# === 5. 儲存模型，之後推論用 ===
joblib.dump(clf, "bge_m3_linear_svc_kocfkos.joblib")
print("\n模型已儲存為 bge_m3_linear_svc_kocfkos.joblib")




'''
# === 6. Logistic Regression 比較 ===
log_clf = LogisticRegression(
    max_iter=5000,           # 避免收斂太慢
    class_weight="balanced", # 一樣處理類別不均
    multi_class="auto",
    n_jobs=-1                # 用多核心加速
)

print("\n開始訓練 LogisticRegression...")
log_clf.fit(X_train, y_train)

y_pred_log = log_clf.predict(X_test)

print("\n=== Classification Report (LogisticRegression, bge-m3) ===")
print(classification_report(y_test, y_pred_log, digits=4))

print("\n=== Confusion Matrix (LogReg) ===")
print(confusion_matrix(y_test, y_pred_log))

joblib.dump(log_clf, "bge_m3_logreg_kocfkos.joblib")
print("\nLogistic Regression 模型已儲存為 bge_m3_logreg_kocfkos.joblib")
'''