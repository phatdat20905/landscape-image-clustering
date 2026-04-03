# src/crawler/crawl_google.py
# ================================================================
#  GOOGLE IMAGES CRAWLER – Selenium + MinIO + MongoDB
#  Đầy đủ: scroll, click, download (http + base64), validate, upload, log
# ================================================================

import os
import sys
import time
import random
import base64
import cv2
import numpy as np
import requests
import argparse
from io import BytesIO
from datetime import datetime
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.config import KEYWORDS, MIN_SIZE, DELAY, MINIO_BUCKET
from src.storage.minio_client import MinioClient
from src.storage.mongodb_client import MongoDBClient

# ================================================================
#  CONFIG
# ================================================================
TARGET = 5000
SCROLL_LIMIT = 15
SCROLL_PAUSE = 2.0
MIN_IMG_DIM = 100
REQUEST_TIMEOUT = 15
DOWNLOAD_RETRIES = 3
UPLOAD_RETRIES = 3

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


# ================================================================
#  SETUP DRIVER
# ================================================================
def setup_driver(headless: bool = True) -> webdriver.Chrome:
    """Khởi tạo Selenium Chrome driver với anti-bot settings"""
    options = Options()
    
    if headless:
        options.add_argument("--headless=new")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--user-agent={USER_AGENT}")
    options.add_argument("--log-level=3")
    options.add_argument("--start-maximized")
    
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # Ẩn webdriver property
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    
    return driver


# ================================================================
#  SCROLL & LOAD MORE IMAGES
# ================================================================
def scroll_and_load(driver: webdriver.Chrome, max_scrolls: int = SCROLL_LIMIT):
    """Scroll trang để load thêm ảnh vào DOM"""
    print("    [Scroll] Loading images...")
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        
        # Nhấn nút "Xem thêm" nếu có
        try:
            more_btn = driver.find_element(
                By.CSS_SELECTOR, 
                "input.mye4qd, button.r0zKGf, button[aria-label*='Show']"
            )
            driver.execute_script("arguments[0].click();", more_btn)
            time.sleep(SCROLL_PAUSE)
        except Exception:
            pass
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        
        last_height = new_height
        scroll_count += 1
    
    print(f"    [Scroll] Done ({scroll_count} scrolls)")


# ================================================================
#  SCRAPE URLS FROM DOM
# ================================================================
def scrape_image_urls_from_dom(driver: webdriver.Chrome) -> list:
    """Lấy tất cả URL ảnh từ các <img> elements trong DOM"""
    print("    [Scrape] Extracting image URLs from DOM...")
    
    urls = []
    try:
        images = driver.find_elements(By.TAG_NAME, "img")
        print(f"    [Scrape] Found {len(images)} img elements")
        
        for idx, img in enumerate(images):
            # Lấy src hoặc data-src
            src = img.get_attribute("src") or img.get_attribute("data-src") or ""
            
            if not src:
                continue
            
            # Bỏ placeholder GIF
            if "data:image/gif" in src or src.startswith("data:image/gif"):
                continue
            
            # Lấy kích thước
            try:
                w = int(img.get_attribute("width") or 0)
                h = int(img.get_attribute("height") or 0)
            except (ValueError, TypeError):
                w, h = 0, 0
            
            # Giữ nếu: không có size attr (có thể ảnh gốc) hoặc size >= MIN_IMG_DIM
            if (w == 0 and h == 0) or (w >= MIN_IMG_DIM and h >= MIN_IMG_DIM):
                urls.append(src)
        
        print(f"    [Scrape] Extracted {len(urls)} URLs")
        return urls
    
    except Exception as e:
        print(f"    [!] Scrape error: {e}")
        return []


# ================================================================
#  DOWNLOAD IMAGE
# ================================================================
def download_image_bytes(
    url: str, 
    session: requests.Session, 
    search_url: str,
    retries: int = DOWNLOAD_RETRIES
) -> bytes:
    """
    Tải ảnh từ URL hoặc base64.
    Retries với exponential backoff.
    """
    
    # ── Base64 ────────────────────────────────────────────────
    if url.startswith("data:image/"):
        try:
            parts = url.split(",", 1)
            if len(parts) == 2:
                encoded = parts[1]
                decoded = base64.b64decode(encoded)
                if len(decoded) > 0:
                    return decoded
        except Exception as e:
            pass
        return b""
    
    # ── HTTP(S) ───────────────────────────────────────────────
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": search_url,
        "Accept": "image/*",
    }
    
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if resp.status_code == 200 and len(resp.content) > 0:
                return resp.content
            
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
        
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
        
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
    
    return b""


# ================================================================
#  VALIDATE IMAGE
# ================================================================
def validate_image_bytes(img_bytes: bytes) -> tuple:
    """
    Validate ảnh từ bytes.
    Trả về (is_valid: bool, width: int, height: int)
    """
    if not img_bytes or len(img_bytes) == 0:
        return False, 0, 0
    
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False, 0, 0
        
        h, w = img.shape[:2]
        
        # Kiểm tra kích thước: ít nhất một chiều >= MIN_SIZE
        max_dim = max(w, h)
        if max_dim < MIN_SIZE:
            return False, w, h
        
        return True, w, h
    
    except Exception as e:
        return False, 0, 0


# ================================================================
#  UPLOAD TO MINIO
# ================================================================
def upload_to_minio(
    minio: MinioClient,
    bucket: str,
    object_name: str,
    img_bytes: bytes,
    retries: int = UPLOAD_RETRIES
) -> bool:
    """Upload ảnh lên MinIO với retry"""
    
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            ok = minio.put_object(
                bucket,
                object_name,
                data=BytesIO(img_bytes),
                length=len(img_bytes),
                content_type="image/jpeg",
            )
            
            if ok:
                return True
        
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2
    
    return False


# ================================================================
#  SAVE TO MONGODB
# ================================================================
def save_to_mongodb(
    mongo: MongoDBClient,
    collection_name: str,
    metadata: dict
) -> bool:
    """Lưu metadata vào MongoDB"""
    try:
        col = mongo.get_col(collection_name)
        col.insert_one(metadata)
        return True
    except Exception as e:
        print(f"      [!] MongoDB insert error: {e}")
        return False


# ================================================================
#  MAIN CRAWL FUNCTION
# ================================================================
def crawl_google_images(
    minio: MinioClient,
    mongo: MongoDBClient,
    target: int = TARGET,
    headless: bool = True
):
    """Crawl ảnh từ Google Images"""
    
    # Setup MongoDB
    col = mongo.get_col("raw")
    mongo.create_indexes(col)
    
    existing = col.count_documents({"source": "google"})
    print(f"\n[MongoDB] Already have {existing} Google images")
    print(f"[Target] {target} images\n")
    
    # Setup Selenium driver
    driver = setup_driver(headless=headless)
    
    # Setup requests session
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    
    saved_count = existing
    
    try:
        for kw_idx, keyword in enumerate(KEYWORDS):
            if saved_count >= target:
                print(f"\n[Target] Reached {saved_count} images. Stopping.")
                break
            
            print(f"\n[{kw_idx+1}/{len(KEYWORDS)}] Keyword: '{keyword} landscape'")
            
            # ── Open Google Images search ──────────────────────
            search_url = (
                f"https://www.google.com/search"
                f"?q={keyword.replace(' ', '+')}+landscape&tbm=isch"
            )
            
            driver.get(search_url)
            time.sleep(3)
            
            # ── Close cookies/banners ──────────────────────────
            for btn_text in ["Accept all", "Reject all", "Accept", "I agree"]:
                try:
                    btn = driver.find_element(
                        By.XPATH,
                        f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{btn_text.lower()}')]"
                    )
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(1)
                    break
                except Exception:
                    pass
            
            # ── Scroll & load images ───────────────────────────
            scroll_and_load(driver, max_scrolls=SCROLL_LIMIT)
            
            # ── Scrape all image URLs ──────────────────────────
            all_urls = scrape_image_urls_from_dom(driver)
            
            if not all_urls:
                print(f"    [!] No URLs found. Skipping.")
                continue
            
            print(f"    [URLs] Processing {len(all_urls)} URLs...")
            
            downloaded = set()
            kw_count = 0
            
            # ── Process each URL ───────────────────────────────
            for url_idx, url in enumerate(all_urls):
                if saved_count >= target:
                    break
                
                if url in downloaded:
                    continue
                
                print(f"      [{url_idx+1}/{len(all_urls)}] ", end="")
                
                # Download
                img_bytes = download_image_bytes(url, session, search_url)
                if not img_bytes:
                    print("[SKIP] download failed")
                    continue
                
                print(f"({len(img_bytes)} bytes) ", end="")
                
                # Validate
                is_valid, w, h = validate_image_bytes(img_bytes)
                if not is_valid:
                    print(f"[SKIP] size {w}x{h} < {MIN_SIZE} (min)")
                    continue
                
                print(f"({w}x{h}) ", end="")
                
                # Upload MinIO
                filename = f"{saved_count:06d}.jpg"
                object_name = f"raw/images/{filename}"
                
                uploaded = upload_to_minio(minio, MINIO_BUCKET, object_name, img_bytes)
                if not uploaded:
                    print("[SKIP] minio failed")
                    continue
                
                print("[OK] minio ", end="")
                
                # Save MongoDB
                metadata = {
                    "filename": filename,
                    "object_name": object_name,
                    "source": "google",
                    "url": url if url.startswith("http") else None,
                    "keyword": keyword,
                    "description": f"{keyword} landscape",
                    "width": w,
                    "height": h,
                    "crawled_at": datetime.now().isoformat(),
                }
                
                saved = save_to_mongodb(mongo, "raw", metadata)
                if not saved:
                    print("[SKIP] mongo failed")
                    continue
                
                print("[OK] mongo SAVED")
                
                downloaded.add(url)
                saved_count += 1
                kw_count += 1
            
            print(f"    [Summary] {keyword}: +{kw_count} images (total: {saved_count}/{target})")
            
            # Random delay between keywords (avoid detection)
            if kw_idx < len(KEYWORDS) - 1:
                delay = DELAY * 2 + random.uniform(3, 7)
                print(f"    [Delay] Waiting {delay:.1f}s before next keyword...")
                time.sleep(delay)
    
    finally:
        driver.quit()
        print("\n[Done] ChromeDriver closed.")
    
    print(f"\n{'='*60}")
    print(f"  CRAWL COMPLETE")
    print(f"  Total saved: {saved_count} images")
    print(f"  MinIO bucket: {MINIO_BUCKET}")
    print(f"  MongoDB collection: raw")
    print(f"{'='*60}\n")


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  GOOGLE IMAGES CRAWLER - SELENIUM 2026")
    print("=" * 70)
    
    parser = argparse.ArgumentParser(
        description="Crawl images from Google Images using Selenium"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET,
        help=f"Number of images to crawl (default: {TARGET})"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show Chrome window (non-headless)"
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Crawl single keyword instead of all"
    )
    
    args = parser.parse_args()
    
    # Override KEYWORDS if single keyword specified
    if args.keyword:
        KEYWORDS = [args.keyword]
    
    minio = MinioClient()
    mongo = MongoDBClient()
    
    try:
        crawl_google_images(
            minio,
            mongo,
            target=args.target,
            headless=not args.show
        )
    except KeyboardInterrupt:
        print("\n[!] Crawl interrupted by user")
    except Exception as e:
        print(f"\n[!] Crawl error: {e}")
        import traceback
        traceback.print_exc()