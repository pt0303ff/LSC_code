import json
import math
import re
import joblib
import pandas as pd
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from FlagEmbedding import BGEM3FlagModel
from typing import Set

# ============ 0) 你的既有資源 ============
CSV_PATH = "1130moli_comments_parsed4_noL.csv"
SVM_PATH = "bge_m3_linear_svc_kocfkos.joblib"

# 欄位（你已確認）
COL_USER = "user"
COL_TIME = "time"
COL_TEXT = "comment"

# ============ 1) 載入模型（沿用你 bge_test.py 的做法） ============
print("Loading SVM model...")
svm_model = joblib.load(SVM_PATH)

print("Loading BGE-M3 embedding model...")
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")  # 沒GPU可改 cpu

def _sigmoid(x: float) -> float:
    # 將 decision score 壓到 0~1 當作信心參考（不是機率，只是可用的尺度）
    return 1.0 / (1.0 + math.exp(-x))

def svm_predict(text: str) -> Dict[str, Any]:
    emb = embed_model.encode([text])["dense_vecs"]
    pred = svm_model.predict(emb)[0]
    score = svm_model.decision_function(emb)

    # decision_function 可能回 array
    if hasattr(score, "ndim") and score.ndim == 2:
        # 取出該筆樣本的各類分數
        row = score[0]  # shape: (n_classes,)
        classes = list(svm_model.classes_)  # 例如 ["KOC","KOF","KOS"] (順序依你訓練)
        pred_idx = classes.index(pred)
        score_val = float(row[pred_idx])
    else:
        # binary 或其他情況
        score_val = float(score[0]) if hasattr(score, "__len__") else float(score)

    conf = _sigmoid(score_val)
    return {"svm_role": pred, "svm_score": score_val, "svm_conf": conf}

# ============ 2) 時間字串轉秒（00:00:08 -> 8） ============
def time_to_seconds(t: str) -> int:
    t = str(t).strip()
    # 支援 HH:MM:SS 或 MM:SS
    parts = t.split(":")
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    if len(parts) == 2:
        mm, ss = parts
        return int(mm) * 60 + int(ss)
    # 其他格式就盡量抓數字
    m = re.findall(r"\d+", t)
    return int(m[-1]) if m else 0

# ============ 3) 每個 user 的狀態 ============
@dataclass
class HistItem:
    t: str
    sec: int
    text: str
    svm_role: str
    svm_conf: float

@dataclass
class UserState:
    current_role: Optional[str] = None
    history: List[HistItem] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    counted_roles: set = field(default_factory=set)     #user已經被統計過

# ============ 4) Ollama / 本地 LLM 設定 ============
OLLAMA_URL = "http://163.18.42.227:11434/api/chat"
OLLAMA_MODEL = "gpt-oss:latest"  # 你可換 taide / qwen 等

COUNTABLE_ROLES = {"KOF", "KOC", "KOS"}   # 要統計的人數角色
SVM_CONF_THRESHOLD = 0.80                 # 低於這個值才呼叫 LLM

ROLE_DEFS = {
    "KOF": "追隨/互動/問候/閒聊（不含明確購買意圖）",
    "KOC": "購買/下單/加一/想買/結帳/詢價/要買等交易意圖",
    "KOS": "有買過商品回來分享使用心得/開箱文/分享宣傳直播",
    "other": "其他"
}

def call_ollama_dynamic(
    user_id: str,
    new_t: str,
    new_sec: int,
    new_text: str,
    current_role: Optional[str],
    history: List[HistItem],
    svm_role: str,
    svm_conf: float,
    history_window: int = 6
) -> Dict[str, Any]:
    # 只取最近 N 則
    hist = history[-history_window:]
    hist_lines = "\n".join(
        [f"- {h.t} | {h.text} | SVM={h.svm_role}({h.svm_conf:.2f})" for h in hist]
    ) or "(無)"

    system = (
        "你是一個直播留言『動態角色識別器』。"
        "你要根據同一使用者的歷史留言，判斷本次留言的角色，並判斷是否發生角色轉移。"
        "你必須只輸出 JSON，不要輸出任何多餘文字。"
    )

    user_prompt = f"""
【角色定義】
{json.dumps(ROLE_DEFS, ensure_ascii=False)}

【使用者資訊】
user_id={user_id}
current_role={current_role}

【歷史留言（最近幾則）】
{hist_lines}

【本次留言】
time={new_t}
seconds={new_sec}
text={new_text}

【SVM 建議】
svm_role={svm_role}
svm_conf={svm_conf:.2f}

【輸出 JSON 規格】（只輸出 JSON）
{{
  "role": "KOF|KOC|KOS|other",
  "changed": true/false,
  "from": "KOF|KOC|KOS|other|null",
  "to": "KOF|KOC|KOS|other|null",
  "confidence": 0.0-1.0,
  "reason": "一句中文理由（<=20字）"
}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    return json.loads(content)

# ============ 5) 主流程：SVM 靜態 → LLM 動態 → 輸出轉移 ============
def run(out_csv="transitions2.csv", history_window=6):
    df = pd.read_csv(CSV_PATH).dropna(subset=[COL_USER, COL_TIME, COL_TEXT]).copy()
    df["sec"] = df[COL_TIME].apply(time_to_seconds)
    df = df.sort_values([COL_USER, "sec"]).reset_index(drop=True)

    states: Dict[str, UserState] = {}
    transitions: List[Dict[str, Any]] = []
    role_counts = {"KOF": 0, "KOC": 0, "KOS": 0}    #統計直播角色人數

    for row in df.itertuples(index=False):
        user = getattr(row, COL_USER)
        t = getattr(row, COL_TIME)
        sec = getattr(row, "sec")
        text = getattr(row, COL_TEXT)

        st = states.setdefault(user, UserState())

       # 1) 先跑 SVM（靜態）
        svm_out = svm_predict(text)
        svm_role = svm_out["svm_role"]
        svm_conf = svm_out["svm_conf"]

        # 2) SVM 高信心就直接採用；低信心才交給 LLM
        if svm_conf >= SVM_CONF_THRESHOLD:
            final_role = svm_role
            llm_out = {
                "role": final_role,
                "changed": (st.current_role is not None and final_role != st.current_role),
                "from": st.current_role,
                "to": final_role,
                "confidence": float(svm_conf),
                "reason": f"SVM高信心直接採用({svm_conf:.2f})"
            }
        else:
            try:
                llm_out = call_ollama_dynamic(
                    user_id=user,
                    new_t=t,
                    new_sec=sec,
                    new_text=text,
                    current_role=st.current_role,
                    history=st.history,
                    svm_role=svm_role,
                    svm_conf=svm_conf,
                    history_window=history_window
                )
                final_role = llm_out["role"]
            except Exception as e:
                # LLM 掛了就 fallback 回 SVM
                llm_out = {
                    "role": svm_role,
                    "changed": (st.current_role is not None and svm_role != st.current_role),
                    "from": st.current_role,
                    "to": svm_role,
                    "confidence": float(svm_conf),
                    "reason": f"LLM不可用，回退SVM：{type(e).__name__}"
                }
                final_role = svm_role

        # 3) 記錄角色轉移
        if llm_out.get("changed") is True:
            evt = {
                "user": user,
                "time": t,
                "sec": sec,
                "from": llm_out.get("from"),
                "to": llm_out.get("to"),
                "text": text,
                "svm_role": svm_role,
                "svm_conf": svm_conf,
                "llm_conf": llm_out.get("confidence"),
                "reason": llm_out.get("reason"),
            }
            transitions.append(evt)
            st.transitions.append(evt)

        # 4) 角色人數統計：同一個 user 在同一個角色只計一次
        if final_role in COUNTABLE_ROLES and final_role not in st.counted_roles:
            role_counts[final_role] += 1
            st.counted_roles.add(final_role)

        # 5) 更新 state
        st.current_role = final_role
        st.history.append(HistItem(t=t, sec=sec, text=text, svm_role=svm_role, svm_conf=svm_conf))

        pd.DataFrame(transitions).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 輸出直播角色統計
    pd.DataFrame([role_counts]).to_csv("role_counts.csv", index=False, encoding="utf-8-sig")

    # 輸出每個 user 曾出現過哪些角色
    user_summary = []
    for u, st in states.items():
        user_summary.append({
            "user": u,
            "final_role": st.current_role,
            "counted_roles": ",".join(sorted(st.counted_roles))
        })
    pd.DataFrame(user_summary).to_csv("user_roles_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Done. transitions saved to: {out_csv}")
    print(f"Total transitions: {len(transitions)}")
    print("Role counts:", role_counts)
    print("Saved: role_counts.csv, user_roles_summary.csv")

print(svm_predict("午安"))
print(svm_predict("我要下單+1"))
if __name__ == "__main__":
    run(out_csv="transitions2.csv", history_window=6)