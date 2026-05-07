import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_llm_intervention(
    gold_csv="0313moli_comments_parsed4_noL.csv",
    pred_csv="all_predictions62.csv",
    out_csv="llm_intervention_eval_result62.csv",
    summary_csv="llm_intervention_eval_summary62.csv",
    user_col="user",
    time_col="time",
    gold_label_col="roles",
    svm_label_col="svm_role",
    pred_label_col="final_role"
):
    """
    評估 LLM 被呼叫介入的樣本中：
    1. 原始 SVM 在這批低信心樣本上的正確率
    2. LLM 輔助後 final_role 的正確率
    3. LLM 輔助後相較於 SVM 的改善幅度
    """

    gold_df = pd.read_csv(gold_csv)
    pred_df = pd.read_csv(pred_csv)

    # =========================
    # 1. 檢查必要欄位
    # =========================
    required_gold_cols = [user_col, time_col, gold_label_col]
    required_pred_cols = [user_col, time_col, "used_llm", svm_label_col, pred_label_col]

    for col in required_gold_cols:
        if col not in gold_df.columns:
            raise ValueError(f"人工標註檔缺少欄位：{col}")

    for col in required_pred_cols:
        if col not in pred_df.columns:
            raise ValueError(f"預測結果檔缺少欄位：{col}")

    # =========================
    # 2. 統一欄位型態
    # =========================
    gold_df[user_col] = gold_df[user_col].astype(str)
    gold_df[time_col] = gold_df[time_col].astype(str)
    gold_df[gold_label_col] = gold_df[gold_label_col].astype(str).str.strip()

    pred_df[user_col] = pred_df[user_col].astype(str)
    pred_df[time_col] = pred_df[time_col].astype(str)
    pred_df[svm_label_col] = pred_df[svm_label_col].astype(str).str.strip()
    pred_df[pred_label_col] = pred_df[pred_label_col].astype(str).str.strip()

    # used_llm 可能是 True/False 或字串
    pred_df["used_llm"] = (
        pred_df["used_llm"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # =========================
    # 3. 只取 LLM 有介入的資料
    # =========================
    llm_df = pred_df[pred_df["used_llm"] == True].copy()

    if len(llm_df) == 0:
        print("沒有任何 LLM 介入樣本，無法評估。")
        return None

    # =========================
    # 4. 與人工標註資料用 user + time 對齊
    # =========================
    merged = pd.merge(
        llm_df,
        gold_df[[user_col, time_col, gold_label_col]],
        on=[user_col, time_col],
        how="left"
    )

    # 移除沒有人工標註答案的資料
    merged = merged.dropna(subset=[gold_label_col]).copy()

    if len(merged) == 0:
        print("LLM 介入樣本沒有成功對齊人工標註資料，請檢查 user/time 格式。")
        return None

    # =========================
    # 5. 計算 SVM 與 LLM-final 是否正確
    # =========================
    merged["svm_correct"] = merged[svm_label_col] == merged[gold_label_col]
    merged["llm_final_correct"] = merged[pred_label_col] == merged[gold_label_col]

    svm_correct_count = int(merged["svm_correct"].sum())
    llm_correct_count = int(merged["llm_final_correct"].sum())

    svm_acc_on_llm_cases = merged["svm_correct"].mean()
    llm_acc_on_llm_cases = merged["llm_final_correct"].mean()
    improvement = llm_acc_on_llm_cases - svm_acc_on_llm_cases

    y_true = merged[gold_label_col]
    y_pred_svm = merged[svm_label_col]
    y_pred_llm = merged[pred_label_col]

    # =========================
    # 6. 輸出主要結果
    # =========================
    print("\n=== LLM Intervention Evaluation ===")
    print(f"LLM intervention count: {len(merged)}")

    print("\n--- SVM on LLM-intervention cases ---")
    print(f"SVM correct count: {svm_correct_count}")
    print(f"SVM Accuracy on LLM cases: {svm_acc_on_llm_cases:.4f}")

    print("\n--- LLM-assisted final result on same cases ---")
    print(f"LLM-final correct count: {llm_correct_count}")
    print(f"LLM Intervention Accuracy: {llm_acc_on_llm_cases:.4f}")

    print("\n--- Improvement ---")
    print(f"Accuracy Improvement: {improvement:.4f}")
    print(f"Accuracy Improvement percentage points: {improvement * 100:.2f}%")

    # =========================
    # 7. SVM classification report
    # =========================
    print("\n=== Classification Report: SVM on LLM-intervention cases ===")
    print(classification_report(y_true, y_pred_svm, zero_division=0))

    print("\n=== Confusion Matrix: SVM on LLM-intervention cases ===")
    labels_svm = sorted(list(set(y_true) | set(y_pred_svm)))
    cm_svm = confusion_matrix(y_true, y_pred_svm, labels=labels_svm)
    cm_svm_df = pd.DataFrame(
        cm_svm,
        index=[f"true_{label}" for label in labels_svm],
        columns=[f"pred_{label}" for label in labels_svm]
    )
    print(cm_svm_df)

    # =========================
    # 8. LLM-final classification report
    # =========================
    print("\n=== Classification Report: LLM-assisted final result ===")
    print(classification_report(y_true, y_pred_llm, zero_division=0))

    print("\n=== Confusion Matrix: LLM-assisted final result ===")
    labels_llm = sorted(list(set(y_true) | set(y_pred_llm)))
    cm_llm = confusion_matrix(y_true, y_pred_llm, labels=labels_llm)
    cm_llm_df = pd.DataFrame(
        cm_llm,
        index=[f"true_{label}" for label in labels_llm],
        columns=[f"pred_{label}" for label in labels_llm]
    )
    print(cm_llm_df)

    # =========================
    # 9. 輸出逐筆比對結果
    # =========================
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 輸出 SVM confusion matrix
    svm_cm_csv = "svm_on_llm_cases_confusion_matrix.csv"
    cm_svm_df.to_csv(svm_cm_csv, encoding="utf-8-sig")

    # 輸出 LLM confusion matrix
    llm_cm_csv = "llm_intervention_confusion_matrix.csv"
    cm_llm_df.to_csv(llm_cm_csv, encoding="utf-8-sig")

    # =========================
    # 10. 輸出 summary
    # =========================
    summary = pd.DataFrame([{
        "llm_intervention_count": len(merged),

        "svm_correct_count_on_llm_cases": svm_correct_count,
        "svm_accuracy_on_llm_cases": svm_acc_on_llm_cases,

        "llm_final_correct_count": llm_correct_count,
        "llm_intervention_accuracy": llm_acc_on_llm_cases,

        "accuracy_improvement": improvement,
        "accuracy_improvement_percentage_points": improvement * 100
    }])

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved detail result to: {out_csv}")
    print(f"Saved SVM confusion matrix to: {svm_cm_csv}")
    print(f"Saved LLM confusion matrix to: {llm_cm_csv}")
    print(f"Saved summary to: {summary_csv}")

    return summary


if __name__ == "__main__":
    evaluate_llm_intervention(
        gold_csv="0313moli_comments_parsed4_noL.csv",
        pred_csv="all_predictions62.csv",
        out_csv="llm_intervention_eval_result62.csv",
        summary_csv="llm_intervention_eval_summary62.csv",
        user_col="user",
        time_col="time",
        gold_label_col="roles",
        svm_label_col="svm_role",
        pred_label_col="final_role"
    )