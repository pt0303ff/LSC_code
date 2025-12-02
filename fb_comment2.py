import time
import csv
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, sys

from bs4 import BeautifulSoup

# ========= 這裡改成你的帳號 & 直播網址 =========
FB_EMAIL = "oabcabc30@gmail.com"      # TODO: 改成你的 FB 信箱
FB_PASSWORD = "zxcvbnm39181019"           # TODO: 改成你的 FB 密碼
#VIDEO_URL = "https://www.facebook.com/molisaka168168/videos/4070493199832772" #1109摩里沙卡  # TODO:改成目標影片網址
VIDEO_URL = "https://www.facebook.com/niusbaby/videos/811460114981729"  #天后闆妹1105
#VIDEO_URL = "https://www.facebook.com/molisaka168168/videos/1580497086642651" #1130摩里沙卡
# ============================================

SCROLL_ROUNDS = 450      # 捲動幾輪（愈多愈慢但愈完整）。設定高一點可捲更多留言（e.g., 5000+ 輪可捲 60000+ 則）
WAIT_BETWEEN_SCROLL = 0.3  # 每輪等待秒數（可降低以加快捲動速度）


#切換成"所有留言"
def set_comment_sort_all(driver):
    """
    將留言排序從「最相關」切換成「所有留言」/ All comments
    """
    # 先找到「最相關 / Most relevant」這個排序按鈕
    sort_button_xpaths = [
        "//span[contains(text(),'最相關')]/ancestor::div[@role='button']",
        "//span[contains(text(),'Most relevant')]/ancestor::div[@role='button']",
    ]

    sort_button = None
    for xp in sort_button_xpaths:
        try:
            sort_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            break
        except Exception:
            continue

    if not sort_button:
        print("找不到『最相關 / Most relevant』排序按鈕，可能介面不同，暫時跳過切換排序。")
        return

    # 點開排序選單
    try:
        driver.execute_script("arguments[0].click();", sort_button)
        time.sleep(1.5)
    except Exception:
        print("點擊排序按鈕失敗，暫時跳過切換排序。")
        return

    # 在選單裡找「所有留言 / All comments」選項
    all_option_xpaths = [
        #"//span[normalize-space(text())='所有留言']",
        #"//span[normalize-space(text())='All comments']",
        "//span[normalize-space(text())='由新到舊']",
        "//span[normalize-space(text())='From new to old']",
    ]

    for xp in all_option_xpaths:
        try:
            option = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            driver.execute_script("arguments[0].click();", option)
            time.sleep(2)
            #print("✅ 已切換為『所有留言 / All comments』")
            print("✅ 已切換為『由新到舊 / From new to old』")
            return
        except Exception:
            continue

    print("有打開排序選單，但沒有找到『所有留言 / All comments』選項，可能 UI 文案不同。")


#自己判斷要捲多少頁面
def get_current_comment_count(driver) -> int:
    """
    從目前頁面粗略數留言數量，用來判斷是否還有新留言載入。
    為了效能我們只抓 message 文字，不做很重的處理。
    """
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")

    # 這裡沿用你後面會用的結構：找 aria-label 含 Comment/留言 的區塊文字
    # 實務上 FB 會變動，如果之後發現數字怪怪的，我們再微調 selector。
    comment_texts = []

    # 嘗試抓常見的留言節點（這是保守寫法，不會爆掉語法錯誤）
    containers = soup.find_all(
        lambda tag: tag.name == "div"
        and tag.get("aria-label")
        and ("留言" in tag.get("aria-label") or "Comment" in tag.get("aria-label"))
    )
    for c in containers:
        # 內層常見留言文字在 span dir="auto"
        for span in c.find_all("span", attrs={"dir": "auto"}):
            text = span.get_text(strip=True)
            if text:
                comment_texts.append(text)

    # 去重，避免同一留言多份 DOM
    unique = set(comment_texts)
    return len(unique)


#開啟影片頁
def open_video_and_open_comments(driver):
    driver.get(VIDEO_URL)
    time.sleep(8)

    # 有些情況會自動顯示留言，如果已經看到留言可以直接返回
    # 否則嘗試點選「留言 / Comments」按鈕或分頁
    possible_xpaths = [
        # 下方「留言」按鈕（中文字）
        "//span[text()='留言']/ancestor::div[@role='button']",
        # 英文介面「Comments」按鈕
        "//span[text()='Comments']/ancestor::div[@role='button']",
        # 有些 watch 介面是 tab
        "//div[@role='tab']//span[text()='留言']",
        "//div[@role='tab']//span[text()='Comments']",
        # aria-label 版本
        "//div[@aria-label='留言']",
        "//div[@aria-label='Comments']"
    ]

    clicked = False
    for xp in possible_xpaths:
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(3)
            clicked = True
            print("已點擊留言按鈕 / Comments tab")
            break
        except Exception:
            continue

    if not clicked:
        print("找不到明確的留言按鈕，先繼續用整頁捲動方式載入留言（之後可再調整 selector）")

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # 若不想顯示瀏覽器畫面，可改成 headless，但調試時建議先看到畫面
    # options.add_argument("--headless=new")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver


#登入FB
def login_facebook(driver):
    driver.get("https://www.facebook.com/login")
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "email"))
    )
    email_input = driver.find_element(By.ID, "email")
    pass_input = driver.find_element(By.ID, "pass")

    email_input.clear()
    email_input.send_keys(FB_EMAIL)
    pass_input.clear()
    pass_input.send_keys(FB_PASSWORD)
    pass_input.send_keys(Keys.RETURN)

    # 等登入完成（首頁或安全驗證）
    time.sleep(8)
    # 若你有兩步驗證，這裡會停著，請在瀏覽器視窗手動完成驗證後再按 Enter 繼續（可視需要加 input()）。


def expand_all_comments(driver,
                        max_rounds: int = 450,
                        stable_rounds_limit: int = 10):
    """
    展開所有留言與回覆：
    - 不再用留言數當判斷
    - 改成：只要找得到「查看更多留言 / 更多回覆」就一直點＋捲
    - 連續 stable_rounds_limit 輪都沒有任何可點按鈕，就停
    """
    open_video_and_open_comments(driver)
    set_comment_sort_all(driver)

    stable_rounds = 0

    for i in range(max_rounds):
        print(f"Scroll round {i+1}/{max_rounds}")

        clicks_this_round = 0

        # 1) 「查看更多留言 / 顯示更多留言 / View more comments」
        more_comment_xpaths = [
            "//span[contains(text(),'查看更多留言')]",
            "//span[contains(text(),'顯示更多留言')]",
            "//span[normalize-space(text())='更多留言']",
            "//span[contains(text(),'View more comments')]",
            "//span[contains(text(),'See more comments')]",
            "//span[contains(text(),'View previous comments')]",
        ]
        for xp in more_comment_xpaths:
            buttons = driver.find_elements(By.XPATH, xp)
            for btn in buttons:
                try:
                    if btn.is_displayed():
                        driver.execute_script("arguments[0].click();", btn)
                        clicks_this_round += 1
                        time.sleep(0.25)
                except Exception:
                    pass

        # 2) 「更多回覆 / View more replies」
        more_reply_xpaths = [
            "//span[contains(text(),'更多回覆')]",
            "//span[contains(text(),'查看回覆')]",
            "//span[contains(text(),'View more replies')]",
            "//span[contains(text(),'See more replies')]",
        ]
        for xp in more_reply_xpaths:
            buttons = driver.find_elements(By.XPATH, xp)
            for btn in buttons:
                try:
                    if btn.is_displayed():
                        driver.execute_script("arguments[0].click();", btn)
                        clicks_this_round += 1
                        time.sleep(0.2)
                except Exception:
                    pass

        # 3) 捲動留言區（若存在），否則捲整頁
        try:
            comments_region = driver.find_element(
                By.XPATH,
                "//div[@role='region' and (contains(@aria-label,'留言') or contains(@aria-label,'Comments'))]"
            )
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight;",
                comments_region
            )
        except Exception:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        time.sleep(WAIT_BETWEEN_SCROLL)

        if clicks_this_round == 0:
            stable_rounds += 1
            print(f"  沒有新的『查看更多/更多回覆』按鈕 (stable {stable_rounds}/{stable_rounds_limit})")
        else:
            stable_rounds = 0

        # 連續多輪都沒有任何可以點的按鈕 → 視為到底
        if stable_rounds >= stable_rounds_limit:
            print("看起來已經沒有可點的查看更多/更多回覆按鈕，停止捲動。")
            break

    time.sleep(3)







def extract_comments_from_page(driver) -> List[Dict]:
    """
    從留言區直接用 Selenium 抓所有留言（不依賴 BeautifulSoup）。
    適用於 Facebook Watch / Live Replay 右側留言面板。
    """

    # 嘗試找留言容器
    try:
        region = driver.find_element(
            By.XPATH,
            "//div[@role='region' and (contains(@aria-label,'留言') or contains(@aria-label,'Comments'))]"
        )
        print("找到留言主容器。")
    except Exception:
        print("找不到留言容器，改用整頁搜尋。")
        region = driver.find_element(By.TAG_NAME, "body")

    # 找所有留言節點（帶 aria-label 或 data-ad-preview='message'）
    comment_nodes = region.find_elements(
        By.XPATH,
        ".//div[@data-ad-preview='message'] | "
        ".//div[@aria-label='留言內容'] | "
        ".//div[@role='article'] | "
        ".//div[contains(@class, 'x1iorvi4') and contains(@class,'x1pi30zi')]"
    )

    print(f"偵測到 {len(comment_nodes)} 個可能的留言節點，準備擷取文字...")

    comments = []
    seen = set()
    for node in comment_nodes:
        try:
            text = node.text.strip()
        except Exception:
            continue

        if not text:
            continue
        # 過濾明顯不是留言的東西
        if text in ("留言", "所有留言", "最相關", "最新",
                    "All comments", "Most relevant", "Newest"):
            continue
        if len(text) < 2:
            continue
        if text.endswith("則留言") or "查看更多留言" in text:
            continue

        if text not in seen:
            seen.add(text)
            comments.append({"message": text})

    print(f"Extracted {len(comments)} unique comments from page.")
    return comments








def save_to_csv(records: List[Dict], filename: str = "fb_live_comments9.csv"):
    if not records:
        print("No comments to save.")
        return
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["message"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Saved {len(records)} comments to {filename}")


if __name__ == "__main__":
    driver = init_driver()
    try:
        print("Logging into Facebook...")
        login_facebook(driver)
        print("Opening video & expanding comments...")
        expand_all_comments(driver)
        print("Extracting comments from loaded page...")
        comments = extract_comments_from_page(driver)
        save_to_csv(comments)
    finally:
        driver.quit()
