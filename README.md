# NIFTY 50 Next-Day Price Forecast

A full-stack educational web application that forecasts the next trading day's NIFTY 50 closing value. It fetches current market data from Yahoo Finance, calculates technical and macroeconomic features, trains a reproducible Random Forest model, and displays the forecast with supporting charts.

> This project is for educational use only. It is not financial or investment advice.

## What the dashboard shows

- Predicted NIFTY 50 next-trading-day close
- Latest available NIFTY 50 close and estimated percentage change
- Historical validation MAE (mean absolute error)
- 120-trading-day NIFTY 50 trend
- Holdout backtest: actual close versus model estimate
- Relative feature importance from the trained model
- Live financial-news sentiment and recent headlines

## Live data sources

The API uses Yahoo Finance through `yfinance` to download roughly five years of daily market history for:

- NIFTY 50 (`^NSEI`) - prediction target
- VIX (`^VIX`)
- S&P 500 (`^GSPC`)
- Nasdaq (`^IXIC`)
- Gold futures (`GC=F`)
- Crude oil futures (`CL=F`)
- USD/INR (`INR=X`)

Financial-news headlines are read from Economic Times and Moneycontrol RSS feeds and scored with VADER. Sentiment is displayed as a live dashboard indicator; it is not a model-training feature, so the prediction pipeline does not depend on Kaggle data or a Kaggle account.

## Model and features

The API trains a `RandomForestRegressor` using time-ordered historical data. The target is the following trading day's NIFTY 50 close.

Features include:

- 1-day and 5-day returns
- 10-day and 30-day moving-average ratios
- RSI (14)
- MACD
- 20-day volatility
- Volume change
- Daily percentage changes in the six macro-market series

The final 20% of the time series is held out for the displayed MAE, avoiding random data splits that can leak future information into a time-series evaluation.

## Performance behavior

- A single batched Yahoo Finance request collects the NIFTY 50 and macro context.
- Successful forecasts are held in memory for 30 minutes.
- Concurrent duplicate requests are serialized, preventing repeated model training.
- Yahoo Finance requests have retry and rate-limit messaging.
- News sentiment refreshes in the background every 30 minutes.

The in-memory cache is reset whenever the service restarts. On Render Free, a service sleeps after inactivity, so the first visitor after a sleep can experience a cold start. A paid always-on instance is required to avoid that hosting-level behavior.

## Run locally

Use Python 3.12 on Windows.

```powershell
cd "C:\path\to\stock-forecast-api"
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

## API endpoints

- `GET /` - dashboard
- `GET /health` - application health check
- `GET /api/forecast?symbol=^NSEI` - live next-day NIFTY 50 forecast
- `GET /api/stocks` - supported instrument list (currently NIFTY 50 only)
- `GET /api/sentiment` - cached financial-news sentiment

## Deploy on Render

1. Push the project files to a GitHub repository with `app/`, `Dockerfile`, `requirements.txt`, and `render.yaml` at the repository root.
2. In Render, create a **Web Service** (or deploy the Blueprint).
3. Choose the Docker runtime. No custom environment variables are required.
4. Render uses `GET /health` as the configured health check and creates a public `onrender.com` URL.

The Dockerfile reads Render's assigned `PORT` automatically.

## Project structure

```text
app/
  main.py                 # FastAPI API, live-data pipeline, model, cache
  static/
    index.html            # Dashboard markup
    app.js                # API calls and Chart.js visualizations
    styles.css            # Responsive dashboard styling
requirements.txt
Dockerfile
render.yaml
```
