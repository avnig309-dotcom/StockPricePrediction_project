"""Live, cache-aware stock forecasting API."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from xgboost import XGBRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from curl_cffi import requests
from yfinance.exceptions import YFRateLimitError

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
MARKET_SYMBOLS = {"vix": "^VIX", "sp500": "^GSPC", "nasdaq": "^IXIC", "gold": "GC=F", "crude": "CL=F", "usd_inr": "INR=X"}
SUPPORTED_STOCKS = {
    "^NSEI": "NIFTY 50 (NSE)",
}
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
]
CACHE_SECONDS = 1800
forecast_cache: dict[str, tuple[float, dict]] = {}
sentiment_cache: dict = {"updated_at": None, "score": 0.0, "label": "Neutral", "headlines": []}
lock = asyncio.Lock()
logger = logging.getLogger("stock_forecast")


def _series(history: pd.DataFrame, column: str = "Close") -> pd.Series:
    value = history[column]
    return value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value


def get_history(symbol: str) -> pd.DataFrame:
    # 1. Initialize browser-impersonating session
    session = requests.Session(impersonate="chrome")
    
    # 2. Fetch main symbol history with retry/backoff
    stock_only = pd.DataFrame()
    last_error: Exception | None = None
    
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol, session=session)
            stock_only = ticker.history(period="5y", interval="1d", auto_adjust=True)
            if not stock_only.empty:
                break
        except (YFRateLimitError, Exception) as exc:
            last_error = exc
            time.sleep(2 ** (attempt + 1))  # Exponential backoff: 2s, 4s, 8s

    if stock_only.empty:
        if last_error:
            raise last_error
        raise ValueError(f"Yahoo Finance returned no price data for '{symbol}'. Check the ticker and retry in a minute.")

    # Yahoo returns each ticker's index timestamped in ITS OWN exchange's
    # timezone (^NSEI -> Asia/Kolkata, ^VIX/^GSPC/etc -> America/New_York).
    # Reindexing a US-timezone series onto an India-timezone index without
    # normalizing first means every timestamp mismatches (they're not
    # actually equal, just the "same day"), so reindex() below would
    # return an all-NaN column, and the final dropna() then wipes out
    # every row in the merged frame. Stripping tz info and truncating to
    # the calendar date fixes the alignment.
    stock_only.index = stock_only.index.tz_localize(None).normalize()

    close = stock_only[["Close"]].rename(columns={"Close": symbol})
    volume = stock_only[["Volume"]].rename(columns={"Volume": symbol})

    if symbol not in close or close[symbol].dropna().size < 160:
        raise ValueError(f"Not enough daily history for '{symbol}'. Use its Yahoo Finance ticker, e.g. RELIANCE.NS.")

    frame = pd.DataFrame({"close": close[symbol], "volume": volume[symbol]}).dropna()

    # 3. Fetch market benchmarks individually using the same session
    for name, ticker_sym in MARKET_SYMBOLS.items():
        try:
            mkt_ticker = yf.Ticker(ticker_sym, session=session)
            mkt_hist = mkt_ticker.history(period="5y", interval="1d", auto_adjust=True)
            if not mkt_hist.empty and "Close" in mkt_hist:
                mkt_hist.index = mkt_hist.index.tz_localize(None).normalize()
                frame[name] = mkt_hist["Close"].reindex(frame.index).ffill().pct_change()
            else:
                frame[name] = 0.0
        except Exception:
            frame[name] = 0.0

    return frame.ffill().dropna()


def features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["return_1d"] = data.close.pct_change()
    data["return_5d"] = data.close.pct_change(5)
    data["sma_10_ratio"] = data.close / data.close.rolling(10).mean() - 1
    data["sma_30_ratio"] = data.close / data.close.rolling(30).mean() - 1
    delta = data.close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean().replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))
    data["macd"] = data.close.ewm(span=12, adjust=False).mean() - data.close.ewm(span=26, adjust=False).mean()
    data["volatility_20"] = data.return_1d.rolling(20).std()
    data["volume_change"] = data.volume.pct_change()
    data["target"] = data.close.shift(-1)
    required = [column for column in data.columns if column != "target"]
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)


def train_and_predict(symbol: str) -> dict:
    raw = get_history(symbol)
    data = features(raw)
    if data.empty:
        raise ValueError(
            f"Not enough clean historical data to build a forecast for '{symbol}' right now. "
            "This can happen if a data provider hiccup drops a row of macro data — try again shortly."
        )
    columns = [c for c in data.columns if c not in {"close", "volume", "target"}]
    train = data[data.target.notna()].copy()

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
    )
    model.fit(train[columns], train.target)

    latest = data.iloc[[-1]]
    predicted = float(model.predict(latest[columns])[0])
    current = float(latest.close.iloc[0])

    split = int(len(train) * 0.8)
    evaluation = XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
    )
    evaluation.fit(train.iloc[:split][columns], train.iloc[:split].target)
    validation = evaluation.predict(train.iloc[split:][columns])
    mae = float(np.mean(np.abs(validation - train.iloc[split:].target)))
    backtest = train.iloc[split:].copy()

    importance_scores = [float(v) for v in model.feature_importances_]
    importance = sorted(zip(columns, importance_scores), key=lambda item: item[1], reverse=True)[:8]

    return {
        "symbol": symbol.upper(),
        "as_of": latest.index[-1].strftime("%Y-%m-%d"),
        "currency_note": "Currency follows the Yahoo Finance listing.",
        "current_close": round(current, 2),
        "predicted_next_close": round(predicted, 2),
        "expected_change_pct": round((predicted / current - 1) * 100, 2),
        "validation_mae": round(mae, 2),
        "indicators": {
            "rsi_14": round(float(latest.rsi_14.iloc[0]), 2),
            "macd": round(float(latest.macd.iloc[0]), 3),
            "volatility_20d": round(float(latest.volatility_20.iloc[0]) * 100, 2),
        },
        "data_source": "Yahoo Finance",
        "model": "XGBoost (5 years daily history; refit on request cache miss)",
        "charts": {
            "price_history": [
                {"date": date.strftime("%Y-%m-%d"), "close": round(float(close), 2)}
                for date, close in raw.close.tail(120).items()
            ],
            "backtest": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "actual": round(float(actual), 2),
                    "predicted": round(float(prediction), 2),
                }
                for date, actual, prediction in zip(backtest.index, backtest.target, validation)
            ],
            "feature_importance": [
                {"name": name.replace("_", " "), "value": round(float(value) * 100, 1)}
                for name, value in importance
            ],
        },
    }


def refresh_sentiment() -> None:
    analyzer, items = SentimentIntensityAnalyzer(), []
    for url in RSS_FEEDS:
        try:
            for entry in feedparser.parse(url).entries[:20]:
                title = entry.get("title", "").strip()
                if title:
                    items.append((title, analyzer.polarity_scores(title)["compound"]))
        except Exception:
            continue
    score = float(np.mean([x[1] for x in items])) if items else 0.0
    label = "Positive" if score > 0.12 else "Negative" if score < -0.12 else "Neutral"
    sentiment_cache.update(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "score": round(score, 3),
            "label": label,
            "headlines": [x[0] for x in items[:6]],
        }
    )


async def sentiment_worker() -> None:
    while True:
        await asyncio.to_thread(refresh_sentiment)
        await asyncio.sleep(1800)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sentiment_worker())
    yield
    task.cancel()


app = FastAPI(title="Live Stock Forecast API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/sentiment")
def sentiment():
    return sentiment_cache


@app.get("/api/stocks")
def stocks():
    return [{"symbol": symbol, "name": name} for symbol, name in SUPPORTED_STOCKS.items()]


@app.get("/api/forecast")
async def forecast(symbol: str = Query(..., min_length=1, max_length=20)):
    key = symbol.strip().upper()
    if key not in SUPPORTED_STOCKS:
        allowed = ", ".join(SUPPORTED_STOCKS)
        raise HTTPException(status_code=400, detail=f"Select one of the supported stocks: {allowed}.")
    cached = forecast_cache.get(key)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return {**cached[1], "cached": True}
    async with lock:
        cached = forecast_cache.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
            return {**cached[1], "cached": True}
        try:
            result = await asyncio.to_thread(train_and_predict, key)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Forecast failed for %s", key)
            if "rate limit" in str(exc).lower() or "too many requests" in str(exc).lower():
                raise HTTPException(
                    status_code=429,
                    detail="Yahoo Finance is rate-limiting requests. Please wait 5–10 minutes before retrying; successful forecasts are cached for 30 minutes.",
                ) from exc
            raise HTTPException(status_code=502, detail=f"Forecast failed: {type(exc).__name__}: {exc}") from exc
        forecast_cache[key] = (time.monotonic(), result)
        return {**result, "cached": False}


@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")