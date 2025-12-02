import pandas as pd
import re

#df = pd.read_csv("fb_live_comments7.csv")
df = pd.read_csv("1130moli_comments.csv")

def normalize_time(t):
    """
    正規化時間格式：
    - mm:ss → 00:mm:ss
    - hh:mm:ss → 保留
    """
    parts = t.split(":")
    if len(parts) == 2:  # mm:ss
        mm, ss = parts
        return f"00:{mm.zfill(2)}:{ss.zfill(2)}"
    elif len(parts) == 3:  # hh:mm:ss
        hh, mm, ss = parts
        return f"{hh.zfill(2)}:{mm.zfill(2)}:{ss.zfill(2)}"
    return t


def parse_message(msg):
    lines = [l.strip() for l in str(msg).split("\n") if l.strip() != ""]
    if len(lines) < 3:
        return None, None, str(msg).strip()

    name = lines[0]
    time = None
    time_idx = 1

    # 找時間：支援 mm:ss 或 hh:mm:ss
    for idx in range(1, min(4, len(lines))):
        m = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', lines[idx])
        if m:
            time = m.group(1)
            time_idx = idx
            break

    if time is None:
        return name, None, ""

    # ⭐ 正規化時間
    time = normalize_time(time)

    # 留言內容
    content_lines = lines[time_idx+1:-1]
    content = "\n".join(content_lines).strip()

    return name, time, content


parsed = df["message"].apply(parse_message)
parsed_df = pd.DataFrame(parsed.tolist(), columns=["user", "time", "comment"])

# 轉換為 datetime 用於排序
parsed_df["time_dt"] = pd.to_datetime(parsed_df["time"], format="%H:%M:%S")

# 排序（小到大）
parsed_df = parsed_df.sort_values(by="time_dt")

# 移除排序欄位後輸出
#parsed_df.drop(columns=["time_dt"]).to_csv("fb_live_comments7_parsed_sorted.csv", index=False)
parsed_df.drop(columns=["time_dt"]).to_csv("1130moli_comments_parsed3.csv", index=False)

print("Parsed comment are saved")
