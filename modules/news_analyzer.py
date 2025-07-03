# modules/news_analyzer.py

import time
import requests
import feedparser
from textblob import TextBlob

class NewsConfig:
    # ใส่โทเค็นของคุณที่นี่
    CRYPTOPANIC_TOKEN = "YOUR_CRYPTOPANIC_TOKEN"
    # URL template สำหรับ RSS feed ของ Yahoo Finance
    YAHOO_FINANCE_RSS_URL = "https://finance.yahoo.com/rss/headline?s={ticker}"
    # อายุ cache (วินาที) ก่อนจะดึงข่าวใหม่
    CACHE_EXPIRY = 7200 

class SmartNewsAnalyzer:
    def __init__(self, config: NewsConfig):
        self.config = config
        # cache format: { symbol: { "ts": timestamp, "data": { sentiment, confidence } } }
        self.cache = {}

    def is_cached(self, symbol: str) -> bool:
        entry = self.cache.get(symbol)
        return bool(entry and time.time() - entry["ts"] < self.config.CACHE_EXPIRY)

    def get_from_cache(self, symbol: str) -> dict:
        return self.cache[symbol]["data"]

    def get_free_sentiment(self, symbol: str) -> dict:
        """
        ดึงคะแนน sentiment + confidence จาก CryptoPanic API
        และ Yahoo Finance RSS feed แล้วรวมกันเป็นผลลัพธ์เดียว
        """
        # ถ้ามีใน cache และยังไม่หมดอายุ ให้คืนค่าเก่า
        if self.is_cached(symbol):
            return self.get_from_cache(symbol)

        # ดึงจากทั้งสองแหล่ง
        cp_list = self._fetch_cryptopanic(symbol)
        yj_list = self._fetch_yahoo_rss(symbol)

        # รวมคะแนนทั้งสองกลุ่ม
        sentiment, confidence = self._aggregate(cp_list + yj_list)

        result = {"sentiment": sentiment, "confidence": confidence}
        # เก็บไว้ใน cache
        self.cache[symbol] = {"ts": time.time(), "data": result}
        return result

    def _fetch_cryptopanic(self, symbol: str) -> list:
        """
        เรียก CryptoPanic API เพื่อดึงข่าวร้อนของเหรียญ
        คืนค่า list ของ (sentiment, confidence)
        """
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.config.CRYPTOPANIC_TOKEN,
            # CryptoPanic ใช้สัญลักษณ์แบบ 'BTC', 'ETH' ฯลฯ
            "currencies": symbol.replace("USDT", ""),
            "filter": "hot"
        }
        try:
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            posts = data.get("results", [])
        except Exception:
            posts = []

        results = []
        for post in posts:
            # รวม title + body เพื่อวิเคราะห์
            text = post.get("title", "") + " " + post.get("body", "")
            s, c = self._analyze_text(text)
            results.append((s, c))
        return results

    def _fetch_yahoo_rss(self, symbol: str) -> list:
        """
        อ่าน RSS feed จาก Yahoo Finance
        คืนค่า list ของ (sentiment, confidence)
        """
        # แปลง 'BTCUSDT' → 'BTC-USD'
        ticker = symbol.replace("USDT", "-USD")
        url = self.config.YAHOO_FINANCE_RSS_URL.format(ticker=ticker)

        try:
            feed = feedparser.parse(url)
            entries = feed.entries
        except Exception:
            entries = []

        results = []
        for entry in entries:
            # ใช้ summary ถ้ามี ไม่งั้นใช้ description หรือเป็นสตริงว่าง
            text = entry.get("summary", entry.get("description", ""))
            s, c = self._analyze_text(text)
            results.append((s, c))
        return results

    def _analyze_text(self, text: str) -> tuple:
        """
        ใช้ TextBlob วิเคราะห์ polarity และ subjectivity
        คืนค่า (polarity, confidence)
        - polarity: -1.0 ถึง +1.0
        - confidence: 0.0 ถึง 1.0 (1 - subjectivity)
        """
        if not text:
            return 0.0, 0.0
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        confidence = max(0.0, 1.0 - subjectivity)
        return polarity, confidence

    def _aggregate(self, items: list) -> tuple:
        """
        รวม list ของ (sentiment, confidence) เป็นตัวเลขเดียว
        - sentiment ถ่วงน้ำหนักด้วย confidence
        - confidence เป็นค่าเฉลี่ย
        """
        if not items:
            return 0.0, 0.0

        total_conf = sum(conf for _, conf in items)
        if total_conf > 0:
            # weighted average sentiment
            sentiment = sum(sent * conf for sent, conf in items) / total_conf
        else:
            sentiment = sum(sent for sent, _ in items) / len(items)
        avg_conf = sum(conf for _, conf in items) / len(items)
        return sentiment, avg_conf
