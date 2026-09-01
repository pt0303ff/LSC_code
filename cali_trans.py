# -*- coding: utf-8 -*-
"""
calibrated_transition_analysis.py

用途：
1. 讀取 dynamic_role3.py / overall_hybrid_eval_result033.csv 的逐筆角色判斷結果。
2. 讀取 0321 後台舊客戶資料，只對「已知舊客戶」補初始 KOC，對「長期觀看未購買」補初始 KOF。
3. 將沒有被標記為舊客戶或長期觀看未購買者的 0321 留言者，補為 other 初始狀態。
4. 依指定 6 個直播階段計算整體與分段角色轉換次數、轉換率並輸出圖表。

注意：
- 本程式不改變原始 SVM-LLMs 判斷流程，只做後處理校正與統計。
- 預設排除直播主帳號「摩里沙卡」，若你要保留，將 EXCLUDE_USERS 改成空集合即可。
"""

import os
import re
from collections import defaultdict
from typing import Optional, Dict, Tuple, Set

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# =========================
# 1. 檔案路徑設定
# =========================

EVENTS_CSV = "./dynamic_result/overall_hybrid_eval_result033.csv"
CUSTOMER_CSV = "./0321_user_role_check/0321 user_final_roles_summary.csv"
OUT_DIR = "./calibrated_transition_output/calibrated_transition_outputs"

EXCLUDE_USERS = {"摩里沙卡"}

COL_USER = "user"
COL_TIME = "time"
COL_SEC = "sec"
COL_TEXT = "text"
COL_ROLE = "final_role"
COL_CUSTOMER_FLAG = "V為舊客戶"

IGNORE_OTHER_FOR_KNOWN_INITIAL_ROLE = True

FOCUS_TRANSITIONS = [
    "other→KOF",
    "other→KOC",
    "KOF→KOC",
    "KOC→KOF",
    "KOC→KOS",
]

ALLOWED_TRANSITIONS = set(FOCUS_TRANSITIONS)

SEGMENTS = [
    {"segment": "S1", "stage_name": "賣商品前互動", "start": "00:00:00", "end": "00:33:37"},
    {"segment": "S2", "stage_name": "第一波銷售", "start": "00:33:38", "end": "01:45:59"},
    {"segment": "S3", "stage_name": "拍賣競標段", "start": "01:46:00", "end": "02:32:59"},
    {"segment": "S4", "stage_name": "品類轉換與需求引導", "start": "02:33:00", "end": "03:16:13"},
    {"segment": "S5", "stage_name": "快速補貨直購", "start": "03:16:14", "end": "03:47:59"},
    {"segment": "S6", "stage_name": "收尾補標", "start": "03:48:00", "end": "04:06:31"},
]

# =========================
# 2. 小工具
# =========================

def time_to_seconds(t) -> int:
    """支援 HH:MM:SS / MM:SS / 秒數。"""
    if pd.isna(t):
        return 0

    if isinstance(t, (int, float)):
        return int(t)

    s = str(t).strip()
    parts = s.split(":")

    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(sec)

        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(sec)

        return int(float(s))

    except Exception:
        nums = re.findall(r"\d+", s)
        return int(nums[-1]) if nums else 0


def normalize_role(role) -> Optional[str]:
    if pd.isna(role):
        return None

    r = str(role).strip()

    if r.lower() in {"", "nan", "none", "null"}:
        return None

    if r.lower() == "other":
        return "other"

    r = r.upper()

    return r if r in {"KOF", "KOC", "KOS"} else str(role).strip()


def assign_segment(sec: int) -> Tuple[Optional[str], Optional[str]]:
    for seg in SEGMENTS:
        if seg["start_sec"] <= sec <= seg["end_sec"]:
            return seg["segment"], seg["stage_name"]

    return None, None


def setup_chinese_font():
    """
    設定圖表字型：
    - 中文優先使用標楷體 DFKai-SB
    - 英文與數字優先使用 Times New Roman
    """

    font_paths = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\kaiu.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    for p in font_paths:
        if os.path.exists(p):
            fm.fontManager.addfont(p)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "DFKai-SB",
        "標楷體",
        "Noto Serif CJK TC",
        "Noto Sans CJK TC",
    ]

    plt.rcParams["axes.unicode_minus"] = False


def add_value_labels_with_rate(ax, values, plot_df):
    """
    長條圖標籤：
    第一行顯示轉換人數
    第二行顯示轉換率與分母
    """
    if not values:
        return

    max_v = max(values) if max(values) > 0 else 1

    for i, v in enumerate(values):
        unique_users = int(plot_df.iloc[i]["unique_users"])
        base_users = int(plot_df.iloc[i]["from_role_base_users"])
        rate = float(plot_df.iloc[i]["conversion_rate_pct"])

        label = f"{unique_users}人\n{rate:.1f}% ({unique_users}/{base_users})"

        ax.text(
            i,
            v + max_v * 0.04,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
            fontname="DFKai-SB",
            clip_on=False,
        )

# =========================
# 3. 讀取與前處理
# =========================

def load_data():
    events = pd.read_csv(EVENTS_CSV)
    customers = pd.read_csv(CUSTOMER_CSV)

    events[COL_USER] = events[COL_USER].astype(str).str.strip()
    customers[COL_USER] = customers[COL_USER].astype(str).str.strip()

    if COL_SEC not in events.columns:
        events[COL_SEC] = events[COL_TIME].apply(time_to_seconds)
    else:
        events[COL_SEC] = events[COL_SEC].apply(time_to_seconds)

    events["_row_order"] = range(len(events))
    events["observed_role"] = events[COL_ROLE].apply(normalize_role)

    events = events.dropna(subset=[COL_USER, COL_SEC, "observed_role"]).copy()

    if EXCLUDE_USERS:
        events = events[~events[COL_USER].isin(EXCLUDE_USERS)].copy()

    initial_role_map = {}

    for row in customers.itertuples(index=False):
        user = str(getattr(row, COL_USER)).strip()
        flag = getattr(row, COL_CUSTOMER_FLAG)
        flag_str = "" if pd.isna(flag) else str(flag).strip()

        if user in EXCLUDE_USERS:
            continue

        if flag_str == "V":
            initial_role_map[user] = "KOC"
        elif "長期" in flag_str and "無購買" in flag_str:
            initial_role_map[user] = "KOF"
        else:
            initial_role_map[user] = "other"

    # 將沒有被標記為舊客戶或長期觀看未購買者的 0321 留言者，補為 other 初始狀態
    all_event_users = set(events[COL_USER].astype(str).str.strip().unique())

    for user in all_event_users:
        if user in EXCLUDE_USERS:
            continue

        if user not in initial_role_map or initial_role_map[user] is None:
            initial_role_map[user] = "other"

    for seg in SEGMENTS:
        seg["start_sec"] = time_to_seconds(seg["start"])
        seg["end_sec"] = time_to_seconds(seg["end"])

    events[["segment", "stage_name"]] = events[COL_SEC].apply(
        lambda x: pd.Series(assign_segment(x))
    )

    events = events.dropna(subset=["segment"]).copy()
    events = events.sort_values([COL_USER, COL_SEC, "_row_order"]).reset_index(drop=True)

    return events, initial_role_map

# =========================
# 4. 角色序列校正與轉換事件建立
# =========================

def build_calibrated_sequences(events, initial_role_map):
    transition_events = []
    sequence_rows = []

    overall_role_base_users: Dict[str, Set[str]] = defaultdict(set)
    segment_role_base_users: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for user, g in events.groupby(COL_USER, sort=False):
        g = g.sort_values([COL_SEC, "_row_order"])

        initial_role = initial_role_map.get(user)
        current_role = initial_role
        has_known_initial = initial_role is not None
        seen_segment = set()

        if initial_role:
            overall_role_base_users[initial_role].add(user)

            sequence_rows.append({
                "user": user,
                "time": "BEFORE_LIVE",
                "sec": None,
                "segment": "PRE",
                "stage_name": "直播前初始角色",
                "text": "後台資料校正",
                "observed_role": None,
                "effective_role": initial_role,
                "source": "known_customer_initial_role",
                "note": "已知舊客戶補KOC；長期觀看未購買補KOF；其餘補other",
            })

        for row in g.itertuples(index=False):
            sec = int(getattr(row, COL_SEC))
            seg = getattr(row, "segment")
            stage_name = getattr(row, "stage_name")
            observed_role = normalize_role(getattr(row, "observed_role"))
            t = getattr(row, COL_TIME)
            text = getattr(row, COL_TEXT) if COL_TEXT in events.columns else ""

            if seg not in seen_segment:
                if current_role is not None:
                    segment_role_base_users[(seg, current_role)].add(user)
                seen_segment.add(seg)

            if observed_role is None:
                continue

            ignored_other = (
                IGNORE_OTHER_FOR_KNOWN_INITIAL_ROLE
                and has_known_initial
                and observed_role == "other"
                and current_role is not None
            )

            if ignored_other:
                effective_role = current_role
                note = "已知初始角色者之other視為無明確本場角色，不觸發轉換"
            else:
                effective_role = observed_role
                note = ""

                if current_role is None:
                    current_role = effective_role

                elif effective_role != current_role:
                    transition_name = f"{current_role}→{effective_role}"

                    # KOS 視為較高階的傳播狀態，不因後續一般互動或購買留言而降回 KOF/KOC
                    if current_role == "KOS" and effective_role in {"KOF", "KOC", "other"}:
                        note = "KOS為累積型傳播角色，後續KOF/KOC/other不視為角色退回"
                        effective_role = current_role

                    else:
                        # 只記錄論文設定的五種轉換
                        if transition_name in ALLOWED_TRANSITIONS:
                            transition_events.append({
                                "user": user,
                                "time": t,
                                "sec": sec,
                                "segment": seg,
                                "stage_name": stage_name,
                                "from_role": current_role,
                                "to_role": effective_role,
                                "transition": transition_name,
                                "observed_role": observed_role,
                                "text": text,
                            })

                        # 即使該轉換不輸出，仍更新目前角色，避免後續角色序列錯亂
                        current_role = effective_role

            if current_role is not None:
                overall_role_base_users[current_role].add(user)
                segment_role_base_users[(seg, current_role)].add(user)

            sequence_rows.append({
                "user": user,
                "time": t,
                "sec": sec,
                "segment": seg,
                "stage_name": stage_name,
                "text": text,
                "observed_role": observed_role,
                "effective_role": current_role,
                "source": "dynamic_role3_output",
                "note": note,
            })

    transitions_df = pd.DataFrame(transition_events)
    sequences_df = pd.DataFrame(sequence_rows)

    return transitions_df, sequences_df, overall_role_base_users, segment_role_base_users

# =========================
# 5. 轉換率表
# =========================

def make_overall_rates(transitions_df, overall_role_base_users):
    if transitions_df.empty:
        return pd.DataFrame(columns=[
            "transition",
            "from_role",
            "to_role",
            "transition_count",
            "unique_users",
            "from_role_base_users",
            "conversion_rate_pct",
            "share_of_all_transitions_pct",
        ])

    total_transitions = len(transitions_df)
    rows = []

    grouped = transitions_df.groupby(["from_role", "to_role", "transition"], dropna=False)

    for (from_role, to_role, transition), g in grouped:
        transition_count = len(g)
        unique_users = g["user"].nunique()
        base = len(overall_role_base_users.get(from_role, set()))

        conv_rate = (unique_users / base * 100) if base else 0
        share = transition_count / total_transitions * 100 if total_transitions else 0

        rows.append({
            "transition": transition,
            "from_role": from_role,
            "to_role": to_role,
            "transition_count": transition_count,
            "unique_users": unique_users,
            "from_role_base_users": base,
            "conversion_rate_pct": conv_rate,
            "share_of_all_transitions_pct": share,
        })

    out = pd.DataFrame(rows).sort_values(
        ["transition_count", "conversion_rate_pct"],
        ascending=False,
    )

    return out


def make_segment_rates(transitions_df, segment_role_base_users):
    rows = []

    for seg in SEGMENTS:
        seg_id = seg["segment"]
        seg_name = seg["stage_name"]

        sub = transitions_df[transitions_df["segment"] == seg_id] if not transitions_df.empty else pd.DataFrame()
        total_transitions = len(sub)

        if sub.empty:
            rows.append({
                "segment": seg_id,
                "stage_name": seg_name,
                "transition": "無轉換",
                "from_role": None,
                "to_role": None,
                "transition_count": 0,
                "unique_users": 0,
                "from_role_base_users": 0,
                "conversion_rate_pct": 0,
                "share_of_segment_transitions_pct": 0,
            })
            continue

        grouped = sub.groupby(["from_role", "to_role", "transition"], dropna=False)

        for (from_role, to_role, transition), g in grouped:
            transition_count = len(g)
            unique_users = g["user"].nunique()
            base = len(segment_role_base_users.get((seg_id, from_role), set()))

            conv_rate = (unique_users / base * 100) if base else 0
            share = transition_count / total_transitions * 100 if total_transitions else 0

            rows.append({
                "segment": seg_id,
                "stage_name": seg_name,
                "transition": transition,
                "from_role": from_role,
                "to_role": to_role,
                "transition_count": transition_count,
                "unique_users": unique_users,
                "from_role_base_users": base,
                "conversion_rate_pct": conv_rate,
                "share_of_segment_transitions_pct": share,
            })

    out = pd.DataFrame(rows).sort_values(
        ["segment", "transition_count"],
        ascending=[True, False],
    )

    return out

# =========================
# 6. 繪圖
# =========================

def filter_plot_df(df, rate_col):
    plot_df = df.copy()

    if "無轉換" in plot_df.get("transition", pd.Series(dtype=str)).values:
        plot_df = plot_df[plot_df["transition"] != "無轉換"].copy()

    if FOCUS_TRANSITIONS is not None and "transition" in plot_df.columns:
        plot_df = plot_df[plot_df["transition"].isin(FOCUS_TRANSITIONS)].copy()

    if plot_df.empty:
        return plot_df

    return plot_df.sort_values([rate_col, "transition_count"], ascending=False)


def plot_overall(overall_rates):
    plot_df = filter_plot_df(overall_rates, "unique_users")

    if plot_df.empty:
        return None

    plot_df = plot_df.head(12)

    labels = plot_df["transition"].tolist()

    # 修改二：整體圖改成畫實際轉換人數，不只畫百分比
    values = plot_df["unique_users"].tolist()

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.bar(labels, values)

    max_v = max(values) if values else 1
    ax.set_ylim(0, max_v * 1.35)

    ax.set_title("整體角色轉換人數與轉換率", fontsize=24, fontname="DFKai-SB", pad=18)
    ax.set_xlabel("角色轉換類型", fontsize=20, fontname="DFKai-SB")
    ax.set_ylabel("轉換人數", fontsize=20, fontname="DFKai-SB", labelpad=18)

    ax.tick_params(axis="x", labelrotation=35, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 修改一：標籤改成顯示人數、轉換率與分母
    add_value_labels_with_rate(ax, values, plot_df)

    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.82,
        bottom=0.28,
    )

    out_path = os.path.join(OUT_DIR, "overall_transition_users_calibrated.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    return out_path


def plot_segments(segment_rates):
    out_paths = []

    for seg in SEGMENTS:
        seg_id = seg["segment"]
        seg_name = seg["stage_name"]

        sub = segment_rates[segment_rates["segment"] == seg_id].copy()
        sub = filter_plot_df(sub, "unique_users")

        fig, ax = plt.subplots(figsize=(11, 7))

        if sub.empty:
            ax.text(
                0.5,
                0.5,
                "此階段無符合條件之角色轉換",
                ha="center",
                va="center",
                fontsize=22,
                fontname="DFKai-SB",
            )

        else:
            sub = sub.head(12)

            labels = sub["transition"].tolist()

            # 修改三：分段圖也改成畫實際轉換人數，不只畫百分比
            values = sub["unique_users"].tolist()

            ax.bar(labels, values)

            max_v = max(values) if values else 1
            ax.set_ylim(0, max_v * 1.35)

            # 修改一：標籤改成顯示人數、轉換率與分母
            add_value_labels_with_rate(ax, values, sub)

            ax.tick_params(axis="x", labelrotation=35, labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax.set_title(f"{seg_id} {seg_name}\n角色轉換人數與轉換率", fontsize=24, fontname="DFKai-SB", pad=18)
        ax.set_xlabel("角色轉換類型", fontsize=20, fontname="DFKai-SB")
        ax.set_ylabel("轉換人數", fontsize=20, fontname="DFKai-SB", labelpad=18)

        fig.subplots_adjust(
            left=0.16,
            right=0.98,
            top=0.78,
            bottom=0.30,
        )

        out_path = os.path.join(OUT_DIR, f"{seg_id}_transition_users_calibrated.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

        out_paths.append(out_path)

    return out_paths

# =========================
# 7. 主程式
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_chinese_font()

    events, initial_role_map = load_data()

    transitions_df, sequences_df, overall_role_base_users, segment_role_base_users = build_calibrated_sequences(
        events,
        initial_role_map,
    )

    overall_rates = make_overall_rates(transitions_df, overall_role_base_users)
    segment_rates = make_segment_rates(transitions_df, segment_role_base_users)

    events.to_csv(
        os.path.join(OUT_DIR, "events_with_segments.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    transitions_df.to_csv(
        os.path.join(OUT_DIR, "calibrated_transition_events.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    sequences_df.to_csv(
        os.path.join(OUT_DIR, "calibrated_user_role_sequences.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    overall_rates.to_csv(
        os.path.join(OUT_DIR, "overall_transition_rates_calibrated.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    segment_rates.to_csv(
        os.path.join(OUT_DIR, "segment_transition_rates_calibrated.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    initial_summary = pd.DataFrame([
        {"initial_role": role, "user_count": len(users)}
        for role, users in overall_role_base_users.items()
    ]).sort_values("user_count", ascending=False)

    initial_summary.to_csv(
        os.path.join(OUT_DIR, "role_base_users_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_overall(overall_rates)
    plot_segments(segment_rates)

    print("完成。輸出資料夾：", OUT_DIR)
    print("有效留言筆數：", len(events))
    print("有效使用者數：", events[COL_USER].nunique())
    print("校正後轉換事件數：", len(transitions_df))

    print("整體轉換率表：")
    print(overall_rates.to_string(index=False))

    print("\n分段轉換率表前 30 筆：")
    print(segment_rates.head(30).to_string(index=False))


if __name__ == "__main__":
    main()