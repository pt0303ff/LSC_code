import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


def evaluate_llm_intervention(
    gold_csv="0321moli_comments_H.csv",
    pred_csv="all_predictions022.csv",  
    out_csv="llm_intervention_eval_result022.csv",  
    summary_csv="llm_intervention_eval_summary022.csv", 
    user_col="user",
    time_col="time",
    gold_label_col="roles_human",
    svm_label_col="svm_role",
    pred_label_col="final_role"
):
    """
    評估兩個層次：

    一、只看 LLM 有介入的樣本
    1. 原始 SVM 在 LLM 介入樣本上的正確率
    2. LLM 輔助後 final_role 在 LLM 介入樣本上的正確率
    3. LLM 輔助後相較於 SVM 的改善幅度

    二、看全部樣本
    1. 全部資料中 SVM 原始結果的整體準確率
    2. 全部資料中混合式架構 final_role 的整體準確率
    3. 混合式架構相較於原始 SVM 的整體改善幅度
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
    # 3. 先與人工標註資料對齊全部樣本
    # =========================
    all_merged = pd.merge(
        pred_df,
        gold_df[[user_col, time_col, gold_label_col]],
        on=[user_col, time_col],
        how="left"
    )

    # 移除沒有人工標註答案的資料
    all_merged = all_merged.dropna(subset=[gold_label_col]).copy()

    if len(all_merged) == 0:
        print("全部預測樣本沒有成功對齊人工標註資料，請檢查 user/time 格式。")
        return None

    # =========================
    # 4. 新增：計算整體 SVM 與整體混合式架構結果
    # =========================
    all_merged["svm_correct_all"] = all_merged[svm_label_col] == all_merged[gold_label_col]
    all_merged["hybrid_correct_all"] = all_merged[pred_label_col] == all_merged[gold_label_col]

    overall_svm_correct_count = int(all_merged["svm_correct_all"].sum())
    overall_hybrid_correct_count = int(all_merged["hybrid_correct_all"].sum())

    overall_svm_acc = all_merged["svm_correct_all"].mean()
    overall_hybrid_acc = all_merged["hybrid_correct_all"].mean()
    overall_improvement = overall_hybrid_acc - overall_svm_acc

    y_true_all = all_merged[gold_label_col]
    y_pred_svm_all = all_merged[svm_label_col]
    y_pred_hybrid_all = all_merged[pred_label_col]

    print("\n=== Overall Evaluation: All Samples ===")
    print(f"Total matched samples: {len(all_merged)}")

    print("\n--- Overall SVM result ---")
    print(f"Overall SVM correct count: {overall_svm_correct_count}")
    print(f"Overall SVM Accuracy: {overall_svm_acc:.4f}")

    print("\n--- Overall Hybrid result: LLM-assisted + remaining SVM ---")
    print(f"Overall Hybrid correct count: {overall_hybrid_correct_count}")
    print(f"Overall Hybrid Accuracy: {overall_hybrid_acc:.4f}")

    print("\n--- Overall Improvement ---")
    print(f"Overall Accuracy Improvement: {overall_improvement:.4f}")
    print(f"Overall Accuracy Improvement percentage points: {overall_improvement * 100:.2f}%")

    print("\n=== Classification Report: Overall SVM ===")
    print(classification_report(y_true_all, y_pred_svm_all, zero_division=0))

    print("\n=== Confusion Matrix: Overall SVM ===")
    labels_svm_all = sorted(list(set(y_true_all) | set(y_pred_svm_all)))
    cm_svm_all = confusion_matrix(y_true_all, y_pred_svm_all, labels=labels_svm_all)
    cm_svm_all_df = pd.DataFrame(
        cm_svm_all,
        index=[f"true_{label}" for label in labels_svm_all],
        columns=[f"pred_{label}" for label in labels_svm_all]
    )
    print(cm_svm_all_df)

    print("\n=== Classification Report: Overall Hybrid result ===")
    print(classification_report(y_true_all, y_pred_hybrid_all, zero_division=0))

    print("\n=== Confusion Matrix: Overall Hybrid result ===")
    labels_hybrid_all = sorted(list(set(y_true_all) | set(y_pred_hybrid_all)))
    cm_hybrid_all = confusion_matrix(y_true_all, y_pred_hybrid_all, labels=labels_hybrid_all)
    cm_hybrid_all_df = pd.DataFrame(
        cm_hybrid_all,
        index=[f"true_{label}" for label in labels_hybrid_all],
        columns=[f"pred_{label}" for label in labels_hybrid_all]
    )
    print(cm_hybrid_all_df)

    # =========================
    # 5. 只取 LLM 有介入的資料
    # =========================
    llm_df = all_merged[all_merged["used_llm"] == True].copy()

    if len(llm_df) == 0:
        print("\n沒有任何 LLM 介入樣本，無法評估 LLM 介入效果。")
        return None

    # =========================
    # 6. 計算 LLM 介入樣本中的 SVM 與 LLM-final 是否正確
    # =========================
    llm_df["svm_correct"] = llm_df[svm_label_col] == llm_df[gold_label_col]
    llm_df["llm_final_correct"] = llm_df[pred_label_col] == llm_df[gold_label_col]

    svm_correct_count = int(llm_df["svm_correct"].sum())
    llm_correct_count = int(llm_df["llm_final_correct"].sum())

    svm_acc_on_llm_cases = llm_df["svm_correct"].mean()
    llm_acc_on_llm_cases = llm_df["llm_final_correct"].mean()
    improvement = llm_acc_on_llm_cases - svm_acc_on_llm_cases

    y_true = llm_df[gold_label_col]
    y_pred_svm = llm_df[svm_label_col]
    y_pred_llm = llm_df[pred_label_col]

    # =========================
    # 7. 輸出 LLM 介入樣本結果
    # =========================
    print("\n=== LLM Intervention Evaluation ===")
    print(f"LLM intervention count: {len(llm_df)}")

    print("\n--- SVM on LLM-intervention cases ---")
    print(f"SVM correct count: {svm_correct_count}")
    print(f"SVM Accuracy on LLM cases: {svm_acc_on_llm_cases:.4f}")

    print("\n--- LLM-assisted final result on same cases ---")
    print(f"LLM-final correct count: {llm_correct_count}")
    print(f"LLM Intervention Accuracy: {llm_acc_on_llm_cases:.4f}")

    print("\n--- LLM Intervention Improvement ---")
    print(f"Accuracy Improvement: {improvement:.4f}")
    print(f"Accuracy Improvement percentage points: {improvement * 100:.2f}%")

    # =========================
    # 8. SVM classification report for LLM cases
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
    # 9. LLM-final classification report for LLM cases
    # =========================
    print("\n=== Classification Report: LLM-assisted final result on LLM-intervention cases ===")
    print(classification_report(y_true, y_pred_llm, zero_division=0))

    print("\n=== Confusion Matrix: LLM-assisted final result on LLM-intervention cases ===")
    labels_llm = sorted(list(set(y_true) | set(y_pred_llm)))
    cm_llm = confusion_matrix(y_true, y_pred_llm, labels=labels_llm)
    cm_llm_df = pd.DataFrame(
        cm_llm,
        index=[f"true_{label}" for label in labels_llm],
        columns=[f"pred_{label}" for label in labels_llm]
    )
    print(cm_llm_df)

    # =========================
    # 10. 輸出逐筆比對結果
    # =========================
    all_out_csv = "overall_hybrid_eval_result033D.csv"   #改檔名
    all_merged.to_csv(os.path.join("dynamic_result", all_out_csv), index=False, encoding="utf-8-sig")

    llm_df.to_csv(os.path.join("dynamic_result", out_csv), index=False, encoding="utf-8-sig")

    # 輸出整體 SVM confusion matrix
    overall_svm_cm_csv = "overall_svm_confusion_matrix033D.csv"  #改檔名
    cm_svm_all_df.to_csv(os.path.join("dynamic_result", overall_svm_cm_csv), encoding="utf-8-sig")

    # 輸出整體混合式架構 confusion matrix
    overall_hybrid_cm_csv = "overall_hybrid_confusion_matrix033D.csv"    #改檔名
    cm_hybrid_all_df.to_csv(os.path.join("dynamic_result", overall_hybrid_cm_csv), encoding="utf-8-sig")

    # 輸出 LLM 介入樣本中的 SVM confusion matrix
    svm_cm_csv = "svm_on_llm_cases_confusion_matrix033D.csv" #改檔名
    cm_svm_df.to_csv(os.path.join("dynamic_result", svm_cm_csv), encoding="utf-8-sig")

    # 輸出 LLM 介入樣本中的 LLM confusion matrix
    llm_cm_csv = "llm_intervention_confusion_matrix033D.csv" #改檔名
    cm_llm_df.to_csv(os.path.join("dynamic_result", llm_cm_csv), encoding="utf-8-sig")

    # =========================
    # 11. 輸出 summary
    # =========================
    summary = pd.DataFrame([{
        "total_matched_samples": len(all_merged),

        "overall_svm_correct_count": overall_svm_correct_count,
        "overall_svm_accuracy": overall_svm_acc,

        "overall_hybrid_correct_count": overall_hybrid_correct_count,
        "overall_hybrid_accuracy": overall_hybrid_acc,

        "overall_accuracy_improvement": overall_improvement,
        "overall_accuracy_improvement_percentage_points": overall_improvement * 100,

        "llm_intervention_count": len(llm_df),

        "svm_correct_count_on_llm_cases": svm_correct_count,
        "svm_accuracy_on_llm_cases": svm_acc_on_llm_cases,

        "llm_final_correct_count": llm_correct_count,
        "llm_intervention_accuracy": llm_acc_on_llm_cases,

        "llm_intervention_accuracy_improvement": improvement,
        "llm_intervention_accuracy_improvement_percentage_points": improvement * 100
    }])

    summary.to_csv(os.path.join("dynamic_result", summary_csv), index=False, encoding="utf-8-sig")

    print(f"\nSaved overall detail result to: {all_out_csv}")
    print(f"Saved LLM intervention detail result to: {out_csv}")
    print(f"Saved overall SVM confusion matrix to: {overall_svm_cm_csv}")
    print(f"Saved overall hybrid confusion matrix to: {overall_hybrid_cm_csv}")
    print(f"Saved SVM confusion matrix on LLM cases to: {svm_cm_csv}")
    print(f"Saved LLM confusion matrix on LLM cases to: {llm_cm_csv}")
    print(f"Saved summary to: {summary_csv}")

    return summary


if __name__ == "__main__":
    evaluate_llm_intervention(
        gold_csv="0321moli_comments_H.csv",
        pred_csv="all_predictions033D.csv",  #改檔案
        out_csv="llm_intervention_eval_result033D.csv",  #改檔案
        summary_csv="llm_intervention_eval_summary033D.csv", #改檔案
        user_col="user",
        time_col="time",
        gold_label_col="roles_human",
        svm_label_col="svm_role",
        pred_label_col="final_role"
    )