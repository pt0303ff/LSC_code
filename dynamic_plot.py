import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager


# =========================
# 0. 字型設定
# =========================
kai_font_path = r"C:\Windows\Fonts\kaiu.ttf"

if os.path.exists(kai_font_path):
    font_manager.fontManager.addfont(kai_font_path)

rcParams["font.family"] = ["Times New Roman", "DFKai-SB"]
rcParams["axes.unicode_minus"] = False

# 全部圖表文字大小統一設定
FONT_SIZE = 25
rcParams["font.size"] = FONT_SIZE
rcParams["axes.titlesize"] = FONT_SIZE
rcParams["axes.labelsize"] = FONT_SIZE
rcParams["xtick.labelsize"] = FONT_SIZE
rcParams["ytick.labelsize"] = FONT_SIZE
rcParams["legend.fontsize"] = FONT_SIZE
rcParams["figure.titlesize"] = FONT_SIZE


# =========================
# 1. 檔案與模型設定
# =========================
# 依照你說的模型對應：
# 011:gpt-oss
# 012:gemma4:e4b
# 013:qwen3.5:9b
#
# 但你目前上傳的檔案名稱是 012、022、032，
# 所以下面用候選檔名方式處理，找到存在的檔案就讀取。

FILES = {
    "gpt-oss": [
        r"dynamic_result\llm_intervention_eval_summary012.csv"
    ],
    "gemma4:e4b": [
        r"dynamic_result\llm_intervention_eval_summary022.csv"
    ],
    "qwen3.5:9b": [
        r"dynamic_result\llm_intervention_eval_summary033.csv"
    ],
}

OUT_DIR = "llm_intervention_charts"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# 2. 讀取檔案函式
# =========================
def find_existing_file(file_candidates):
    for file_path in file_candidates:
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(
        f"找不到檔案，請確認以下其中一個檔案是否存在：{file_candidates}"
    )


summary_rows = []

for model_name, file_candidates in FILES.items():
    file_path = find_existing_file(file_candidates)
    df = pd.read_csv(file_path)

    # 每個 summary 檔只有一列
    row = df.iloc[0]

    total_samples = row["total_matched_samples"]
    intervention_count = row["llm_intervention_count"]
    intervention_rate = intervention_count / total_samples

    summary_rows.append({
        "model": model_name,
        "file": file_path,

        # 整體資料表現
        "total_matched_samples": row["total_matched_samples"],
        "overall_svm_correct_count": row["overall_svm_correct_count"],
        "overall_svm_accuracy": row["overall_svm_accuracy"],
        "overall_hybrid_correct_count": row["overall_hybrid_correct_count"],
        "overall_hybrid_accuracy": row["overall_hybrid_accuracy"],
        "overall_accuracy_improvement": row["overall_accuracy_improvement"],
        "overall_accuracy_improvement_percentage_points": row["overall_accuracy_improvement_percentage_points"],

        # LLM 介入樣本表現
        "llm_intervention_count": row["llm_intervention_count"],
        "svm_correct_count_on_llm_cases": row["svm_correct_count_on_llm_cases"],
        "svm_accuracy_on_llm_cases": row["svm_accuracy_on_llm_cases"],
        "llm_final_correct_count": row["llm_final_correct_count"],
        "llm_intervention_accuracy": row["llm_intervention_accuracy"],
        "llm_intervention_accuracy_improvement": row["llm_intervention_accuracy_improvement"],
        "llm_intervention_accuracy_improvement_percentage_points": row["llm_intervention_accuracy_improvement_percentage_points"],

        # 額外計算
        "intervention_rate": intervention_rate,
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(OUT_DIR, "llm_intervention_summary_all_models.csv"),
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 3. 圖1：整體 SVM vs LLM+SVM 準確率
# =========================
x = np.arange(len(summary_df["model"]))
width = 0.35

plt.figure(figsize=(16, 9))

bars1 = plt.bar(
    x - width / 2,
    summary_df["overall_svm_accuracy"],
    width,
    label="SVM"
)

bars2 = plt.bar(
    x + width / 2,
    summary_df["overall_hybrid_accuracy"],
    width,
    label="LLMs + SVM"
)

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE
        )

plt.xticks(x, summary_df["model"], fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.ylim(0, 1)
plt.ylabel("Accuracy", fontsize=FONT_SIZE)
plt.title("整體 SVM 與 LLMs+SVM 準確率比較", fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "01_overall_svm_vs_hybrid_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# 4. 圖2：只看 LLM 介入樣本的準確率
# =========================
plt.figure(figsize=(16, 9))

bars1 = plt.bar(
    x - width / 2,
    summary_df["svm_accuracy_on_llm_cases"],
    width,
    label="SVM on LLMs cases"
)

bars2 = plt.bar(
    x + width / 2,
    summary_df["llm_intervention_accuracy"],
    width,
    label="LLMs final"
)

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE
        )

plt.xticks(x, summary_df["model"], fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.ylim(0, 1)
plt.ylabel("Accuracy", fontsize=FONT_SIZE)
plt.title("LLMs 介入樣本中 SVM 與 LLMs 判斷準確率比較", fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "02_llm_intervention_cases_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# 5. 圖3：準確率提升幅度比較
# =========================
improvement_metrics = [
    "overall_accuracy_improvement_percentage_points",
    "llm_intervention_accuracy_improvement_percentage_points"
]

improvement_labels = [
    "整體提升百分點",
    "LLMs介入樣本提升百分點"
]

plt.figure(figsize=(16, 9))

bars1 = plt.bar(
    x - width / 2,
    summary_df[improvement_metrics[0]],
    width,
    label=improvement_labels[0]
)

bars2 = plt.bar(
    x + width / 2,
    summary_df[improvement_metrics[1]],
    width,
    label=improvement_labels[1]
)

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE
        )

plt.xticks(x, summary_df["model"], fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.ylabel("Percentage points", fontsize=FONT_SIZE)
plt.title("LLMs 輔助判斷後準確率提升幅度比較", fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "03_accuracy_improvement_percentage_points.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# 6. 圖4：LLM 介入次數
# =========================
plt.figure(figsize=(16, 9))

bars = plt.bar(
    summary_df["model"],
    summary_df["llm_intervention_count"]
)

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 3,
        f"{int(value)}",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE
    )

plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.ylabel("Count", fontsize=FONT_SIZE)
plt.title("各模型 LLMs 介入判斷次數比較", fontsize=FONT_SIZE)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "04_llm_intervention_count.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# 7. 圖5：LLM 介入比例
# =========================
plt.figure(figsize=(16, 9))

bars = plt.bar(
    summary_df["model"],
    summary_df["intervention_rate"]
)

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.1%}",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE
    )

plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.ylim(0, 1)
plt.ylabel("Intervention rate", fontsize=FONT_SIZE)
plt.title("各模型 LLMs 介入比例比較", fontsize=FONT_SIZE)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "05_llm_intervention_rate.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# 8. Hybrid 混淆矩陣圖
# =========================

CONFUSION_FILES = {
    "gpt-oss": r"dynamic_result\overall_hybrid_confusion_matrix012.csv",
    "gemma4:e4b": r"dynamic_result\overall_hybrid_confusion_matrix022.csv",
    "qwen3.5:9b": r"dynamic_result\overall_hybrid_confusion_matrix033.csv",
    "SVM": r"dynamic_result\overall_svm_confusion_matrix033.csv"
}

LABELS = ["KOC", "KOF", "KOS", "other"]

for model_name, cm_file in CONFUSION_FILES.items():
    cm_df = pd.read_csv(cm_file)

    # 取出矩陣數值
    cm = cm_df[["pred_KOC", "pred_KOF", "pred_KOS", "pred_other"]].values

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)

    plt.title(f"Hybrid 混淆矩陣 - {model_name}", fontsize=FONT_SIZE)
    plt.xlabel("模型預測標籤", fontsize=FONT_SIZE)
    plt.ylabel("人工標註標籤", fontsize=FONT_SIZE)

    plt.xticks(np.arange(len(LABELS)), LABELS, fontsize=FONT_SIZE)
    plt.yticks(np.arange(len(LABELS)), LABELS, fontsize=FONT_SIZE)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    # 在格子中標示數字
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=FONT_SIZE
            )

    plt.tight_layout()

    safe_name = (
        model_name
        .replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(OUT_DIR, f"06_hybrid_confusion_matrix_{safe_name}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# 9. 輸出結果
# =========================
print("=== LLM Intervention Summary ===")
print(summary_df.round(4))

print(f"\n圖表與整理後 CSV 已輸出至：{OUT_DIR}")