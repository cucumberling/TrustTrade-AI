from __future__ import annotations
"""
Sentiment Agent: combines two free fundamental sentiment sources.

1. Crypto Fear & Greed Index (alternative.me) — daily 0-100 score
   - 0-25  = Extreme Fear   → contrarian BUY
   - 25-45 = Fear           → mild BUY
   - 45-55 = Neutral
   - 55-75 = Greed          → mild SELL
   - 75-100= Extreme Greed  → contrarian SELL

2. Cointelegraph Bitcoin RSS headlines — last 20 headlines, scored by keyword OR Claude

Output: signal in [-1.0, +1.0]. Negative = bearish sentiment / contrarian bullish.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import re
import urllib.request
import urllib.error
import time

from config.settings import settings
from utils.logger import logger


# Heuristic keyword scoring (used when no LLM key available)
BULLISH_WORDS = {
    "rally", "surge", "soar", "jump", "gain", "high", "ath", "all-time high",
    "bullish", "buy", "accumulate", "breakout", "rebound", "recover", "rise",
    "approval", "approved", "etf inflow", "institutional", "adoption",
}
BEARISH_WORDS = {
    "crash", "plunge", "slump", "drop", "fall", "low", "bearish", "sell-off",
    "liquidat", "dump", "decline", "warning", "fraud", "hack", "ban", "lawsuit",
    "outflow", "rejected", "fear", "panic", "collapse",
}


@dataclass
class SentimentSignal:
    signal: float                       # -1.0 (bearish) to +1.0 (bullish), CONTRARIAN of crowd
    fear_greed_value: int               # 0-100
    fear_greed_label: str               # "Extreme Fear" / "Greed" / etc
    fear_greed_signal: float            # Contrarian signal from F&G
    headline_signal: float              # Sentiment from headlines
    headline_count: int                 # How many headlines analyzed
    used_llm: bool                      # True if Claude was used for scoring
    sample_headlines: list[str] = field(default_factory=list)
    reason: str = ""


class SentimentAgent:
    FNG_URL = "https://api.alternative.me/fng/?limit=7"
    NEWS_URL = "https://cointelegraph.com/rss/tag/bitcoin"

    def __init__(
        self,
        fear_greed_weight: float = 0.7,    # F&G is the more proven signal
        headline_weight: float = 0.3,
        cache_ttl_seconds: int = 1800,     # 30 min cache (sentiment moves slowly)
        use_llm: bool = True,              # Use Claude for headline NLP if API key set
    ):
        self.fear_greed_weight = fear_greed_weight
        self.headline_weight = headline_weight
        self.cache_ttl = cache_ttl_seconds
        self.use_llm = use_llm and bool(settings.llm.api_key)
        self._cache: dict[str, tuple[float, object]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Data fetchers (cached)
    # ──────────────────────────────────────────────────────────────────
    def _cached_fetch(self, key: str, url: str) -> Optional[bytes]:
        now = time.time()
        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts < self.cache_ttl:
                return data  # type: ignore
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TrustTrade-AI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                self._cache[key] = (now, data)
                return data
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning(f"SentimentAgent: fetch failed for {key}: {e}")
            return None

    def _fetch_fear_greed(self) -> Optional[tuple[int, str]]:
        raw = self._cached_fetch("fng", self.FNG_URL)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            entries = data.get("data", [])
            if not entries:
                return None
            latest = entries[0]
            value = int(latest.get("value", 50))
            label = latest.get("value_classification", "Neutral")
            return value, label
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def _fetch_headlines(self, max_items: int = 20) -> list[str]:
        raw = self._cached_fetch("news", self.NEWS_URL)
        if not raw:
            return []
        text = raw.decode("utf-8", errors="ignore")
        # Extract titles only inside <item> blocks (skips <channel><title>)
        items = re.findall(r"<item\b[^>]*>(.*?)</item>", text, flags=re.DOTALL)
        titles: list[str] = []
        for item in items:
            m = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, flags=re.DOTALL)
            if m:
                t = m.group(1).strip()
                if t:
                    titles.append(t)
            if len(titles) >= max_items:
                break
        return titles

    # ──────────────────────────────────────────────────────────────────
    # Sentiment scoring
    # ──────────────────────────────────────────────────────────────────
    def _score_fear_greed(self, value: int) -> float:
        """0=extreme fear → +1 (bullish), 100=extreme greed → -1 (bearish). Linear."""
        # Map [0, 100] → [+1, -1]
        return (50 - value) / 50.0

    def _score_headlines_keywords(self, headlines: list[str]) -> float:
        """Heuristic keyword scoring fallback (no LLM)."""
        if not headlines:
            return 0.0
        bull = bear = 0
        for h in headlines:
            lower = h.lower()
            for w in BULLISH_WORDS:
                if w in lower:
                    bull += 1
                    break
            for w in BEARISH_WORDS:
                if w in lower:
                    bear += 1
                    break
        total = bull + bear
        if total == 0:
            return 0.0
        # Headline sentiment is NOT contrarian — it tracks the crowd directly
        return (bull - bear) / total

    def _score_headlines_llm(self, headlines: list[str]) -> Optional[float]:
        """Score with Claude. Returns None if API call fails."""
        try:
            from anthropic import Anthropic
        except ImportError:
            return None

        try:
            client = Anthropic(api_key=settings.llm.api_key)
            joined = "\n".join(f"- {h}" for h in headlines[:15])
            prompt = (
                "You are a crypto market sentiment analyst. Below are recent Bitcoin news "
                "headlines. Output ONLY a single number between -1.0 and +1.0 representing "
                "the aggregate sentiment, where -1.0 = strongly bearish (price will drop), "
                "0 = neutral, +1.0 = strongly bullish (price will rise). No other text.\n\n"
                f"Headlines:\n{joined}\n\nScore:"
            )
            resp = client.messages.create(
                model=settings.llm.model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Extract first float in the response
            m = re.search(r"-?\d+\.?\d*", text)
            if m:
                score = float(m.group())
                return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"SentimentAgent: LLM scoring failed: {e}")
        return None

    # ──────────────────────────────────────────────────────────────────
    # Main entry
    # ──────────────────────────────────────────────────────────────────
    def analyze(self) -> SentimentSignal:
        # Fear & Greed
        fng = self._fetch_fear_greed()
        if fng:
            fng_value, fng_label = fng
            fng_signal = self._score_fear_greed(fng_value)
        else:
            fng_value, fng_label, fng_signal = 50, "Unknown", 0.0

        # Headlines
        headlines = self._fetch_headlines()
        headline_signal = 0.0
        used_llm = False
        if headlines:
            if self.use_llm:
                llm_score = self._score_headlines_llm(headlines)
                if llm_score is not None:
                    headline_signal = llm_score
                    used_llm = True
                else:
                    headline_signal = self._score_headlines_keywords(headlines)
            else:
                headline_signal = self._score_headlines_keywords(headlines)

        # Combine: F&G is contrarian, headlines track the crowd directly.
        # When F&G says "extreme fear" AND headlines are bearish → strong contrarian buy.
        combined = (
            self.fear_greed_weight * fng_signal +
            self.headline_weight * (-headline_signal)  # invert headlines so both align
        )
        combined = max(-1.0, min(1.0, combined))

        # Build reason
        if fng_value <= 25:
            label_part = f"F&G = {fng_value} (Extreme Fear) → strong contrarian BUY"
        elif fng_value <= 45:
            label_part = f"F&G = {fng_value} (Fear) → mild BUY"
        elif fng_value >= 75:
            label_part = f"F&G = {fng_value} (Extreme Greed) → strong contrarian SELL"
        elif fng_value >= 55:
            label_part = f"F&G = {fng_value} (Greed) → mild SELL"
        else:
            label_part = f"F&G = {fng_value} (Neutral)"
        news_part = (
            f"News sentiment: {headline_signal:+.2f} "
            f"({'LLM' if used_llm else 'keyword'}, {len(headlines)} headlines)"
        )
        reason = f"{label_part}. {news_part}"

        result = SentimentSignal(
            signal=round(combined, 4),
            fear_greed_value=fng_value,
            fear_greed_label=fng_label,
            fear_greed_signal=round(fng_signal, 4),
            headline_signal=round(headline_signal, 4),
            headline_count=len(headlines),
            used_llm=used_llm,
            sample_headlines=headlines[:5],
            reason=reason,
        )

        logger.info(
            f"Sentiment Agent: signal={result.signal}",
            fng=f"{fng_value} ({fng_label})",
            headlines=len(headlines),
            llm=used_llm,
        )

        return result


sentiment_agent = SentimentAgent()
