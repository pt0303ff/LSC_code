import pandas as pd
import re
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# =========================
# 1. 時間轉秒數
# =========================
def time_to_seconds(t):
    """
    支援 HH:MM:SS 或 MM:SS 格式
    """
    t = str(t).strip()
    parts = t.split(":")

    try:
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
        elif len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        else:
            m = re.findall(r"\d+", t)
            return int(m[-1]) if m else 0
    except:
        return 0


# =========================
# 2. 從人工標註資料建立 gold_transitions.csv
# =========================
def build_gold_transitions_from_labeled_csv(
    labeled_csv="0313moli_comments_parsed4_noL.csv",
    out_csv="gold_transitions.csv",
    user_col="user",
    time_col="time",
    text_col="comment",
    label_col="roles"
):
    """
    從人工標註好的逐則留言資料建立 gold transition。
    
    輸入資料至少需要包含：
    user, time, comment, roles
    
    輸出：
    gold_transitions.csv
    """

    df = pd.read_csv(labeled_csv)

    # 移除缺漏資料
    df = df.dropna(subset=[user_col, time_col, text_col, label_col]).copy()

    # 統一型態
    df[user_col] = df[user_col].astype(str)
    df[time_col] = df[time_col].astype(str)
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str).str.strip()

    # 建立秒數欄位並排序
    df["sec"] = df[time_col].apply(time_to_seconds)
    df = df.sort_values([user_col, "sec"]).reset_index(drop=True)

    gold_events = []

    for user, group in df.groupby(user_col):
        previous_role = None

        for _, row in group.iterrows():
            current_role = str(row[label_col]).strip()

            # 若角色空白或 nan 字串，略過
            if current_role == "" or current_role.lower() == "nan":
                continue

            # 發生角色變化才記錄 transition
            if previous_role is not None and current_role != previous_role:
                gold_events.append({
                    "user": row[user_col],
                    "time": row[time_col],
                    "sec": row["sec"],
                    "from": previous_role,
                    "to": current_role,
                    "text": row[text_col],
                    "gold_transition": previous_role + "->" + current_role
                })

            previous_role = current_role

    gold_df = pd.DataFrame(gold_events)

    gold_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n=== Gold Transition Built ===")
    print(f"Input labeled file: {labeled_csv}")
    print(f"Saved gold transitions to: {out_csv}")
    print(f"Total gold transitions: {len(gold_df)}")

    if len(gold_df) > 0:
        print("\nGold transition examples:")
        print(gold_df.head())

    return gold_df


# =========================
# 3. 評估 dynamic_role2.py 輸出的 transitions4.csv
# =========================
def evaluate_transition_accuracy(
    pred_csv="transitions4.csv",
    gold_csv="gold_transitions.csv",
    out_csv="transition_eval_result.csv"
):
    """
    比較模型輸出的 transitions4.csv 與人工標註轉換出的 gold_transitions.csv。
    
    評估指標：
    1. Transition Accuracy
    2. Macro Precision / Recall / F1
    3. Weighted Precision / Recall / F1
    4. Classification Report
    5. Confusion Matrix
    """

    pred_df = pd.read_csv(pred_csv)
    gold_df = pd.read_csv(gold_csv)

    if len(gold_df) == 0:
        print("gold_transitions.csv 沒有任何 transition，無法評估。")
        return None

    # 確保欄位型態一致
    for df in [pred_df, gold_df]:
        df["user"] = df["user"].astype(str)
        df["time"] = df["time"].astype(str)
        df["from"] = df["from"].astype(str).str.strip()
        df["to"] = df["to"].astype(str).str.strip()

    # 模型預測 transition label
    pred_df["pred_transition"] = pred_df["from"] + "->" + pred_df["to"]

    # gold transition label
    if "gold_transition" not in gold_df.columns:
        gold_df["gold_transition"] = gold_df["from"] + "->" + gold_df["to"]

    # 只保留 pred 裡需要的欄位
    keep_cols = ["user", "time", "pred_transition"]

    optional_cols = [
        "text", "svm_role", "svm_conf",
        "llm_conf", "reason"
    ]

    for col in optional_cols:
        if col in pred_df.columns:
            keep_cols.append(col)

    # 用 user + time 對齊
    merged = pd.merge(
        gold_df[["user", "time", "gold_transition", "from", "to", "text"]],
        pred_df[keep_cols],
        on=["user", "time"],
        how="left",
        suffixes=("_gold", "_pred")
    )

    # 若模型沒有預測到該 transition
    merged["pred_transition"] = merged["pred_transition"].fillna("NO_PRED")

    # 是否完全正確
    merged["correct"] = merged["gold_transition"] == merged["pred_transition"]

    y_true = merged["gold_transition"]
    y_pred = merged["pred_transition"]

    transition_accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n=== Transition Evaluation Result ===")
    print(f"Gold transition count: {len(merged)}")
    print(f"Correct transition count: {merged['correct'].sum()}")
    print(f"Transition Accuracy: {transition_accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\n=== Confusion Matrix ===")
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels]
    )

    print(cm_df)

    # 輸出逐筆比對結果
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 輸出 confusion matrix
    cm_out_csv = "transition_confusion_matrix.csv"
    cm_df.to_csv(cm_out_csv, encoding="utf-8-sig")

    # 輸出 summary
    summary = pd.DataFrame([{
        "gold_transition_count": len(merged),
        "correct_transition_count": int(merged["correct"].sum()),
        "transition_accuracy": transition_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }])

    summary_out_csv = "transition_eval_summary.csv"
    summary.to_csv(summary_out_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved detail result to: {out_csv}")
    print(f"Saved confusion matrix to: {cm_out_csv}")
    print(f"Saved summary to: {summary_out_csv}")

    return summary


# =========================
# 4. 主程式
# =========================
if __name__ == "__main__":

    # Step 1：從人工標註資料建立 gold transition
    build_gold_transitions_from_labeled_csv(
        labeled_csv="0313moli_comments_parsed4_noL.csv",
        out_csv="gold_transitions1.csv",
        user_col="user",
        time_col="time",
        text_col="comment",
        label_col="roles"
    )

    # Step 2：讀取 dynamic_role2.py 輸出的 transitions4.csv 進行評估
    evaluate_transition_accuracy(
        pred_csv="transitions4.csv",
        gold_csv="gold_transitions1.csv",
        out_csv="transition_eval_result_gptoss.csv"
    )