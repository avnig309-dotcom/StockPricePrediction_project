# Stock Price Prediction Using Machine Learning

## About the Project

The stock market is shaped by a mix of factors — company performance, global markets, investor sentiment, and plain randomness — which makes it notoriously hard to predict. This project aims at predicting the next day's closing price of the NIFTY 50 index using historical market data, technical indicators, and news sentiment.

Rather than feeding raw prices straight into a model, we build out a richer feature set: moving averages, RSI, MACD, Bollinger Bands, ATR, and — going a step further — a daily sentiment score derived from financial news headlines. The idea is that price action alone only tells half the story; how the market is *talking* about a stock or index on a given day carries information too.

The project follows a full ML workflow: collecting and cleaning the data, engineering features (including sentiment), exploring it visually, training several regression models, and comparing how each one performs.

## Project Objectives

- Predict the next day's closing price of the NIFTY 50 index
- Analyse historical stock market data and identify patterns
- Engineer technical indicators that capture trend, momentum, and volatility
- Incorporate news sentiment as an additional predictive signal
- Explore the data visually to surface trends and relationships
- Train and compare multiple regression models
- Evaluate each model with standard regression metrics

## Datasets

**Price data** — downloaded via the `yfinance` library.
- Primary ticker: `^NSEI` (NIFTY 50)

Since Indian equity markets don't move in isolation from global markets, commodities, and currency movements, six supporting instruments are downloaded the same way and merged into the main dataset by date:

- **India VIX** (`^INDIAVIX`) — reflects expected near-term volatility in Indian equities
- **S&P 500** (`^GSPC`) — tracks U.S. market sentiment, which frequently spills over into Asian markets the next trading session
- **Nasdaq** (`^IXIC`) — heavily tech-weighted, relevant given how much of NIFTY 50's movement is tied to IT and tech-adjacent stocks
- **Gold Futures** (`GC=F`) — a traditional safe-haven asset; gold prices often move inversely to equity risk appetite
- **Crude Oil Futures** (`CL=F`) — India is a major oil importer, so crude price swings directly affect inflation expectations and corporate costs across sectors
- **USD/INR Exchange Rate** (`USDINR=X`) — currency depreciation/appreciation affects import costs, foreign investment flows, and IT/export-sector earnings

Each of these contributes its own `Close` price, merged in as a separate column (e.g., `Close vix`, `Close snp`) alongside the main NIFTY 50 OHLCV data.

Together, the full price dataset contains: Date, Open, High, Low, Close, Adjusted Close, Volume, and the six supporting `Close` columns above, spanning from 2013 through mid-2026 — long enough for a model to pick up on longer-term market behavior, not just short-term noise.

**News data** — since price and volume alone can't capture *why* the market moved on a given day, we pulled in two supplementary datasets from Kaggle:
- **NifSent50**, a dataset of financial headlines tied to NIFTY 50 constituent companies
- **FindKG (GDELT-based)**, a broader financial news and mentions dataset 

Headlines from both sources are combined into a single timeline, sentiment-scored, and merged into the main dataset by date.

## Data Preprocessing

Before anything is modeled, the dataset goes through a cleanup pass:

- Checking for missing values and duplicate records
- Verifying correct data types (particularly making sure date columns are true `datetime` objects, not strings — a subtle but common source of merge failures)
- Confirming the data is sorted in chronological order
- Removing rows with zero trading volume (holidays/non-trading days that slip into the raw download)

Once verified, technical indicators are engineered on top of the cleaned data, and any rows with missing values introduced by rolling-window calculations (e.g., the first 19 days before a 20-day moving average can be computed) are dropped.

## Feature Engineering

### Trend Indicators
- **SMA (20-Day)** — smooths out short-term noise to reveal the underlying trend
- **SMA (50-Day)** — a longer lens for spotting sustained trends
- **EMA (20-Day)** — weights recent prices more heavily, so it reacts faster to fresh trend changes than a simple average would
- **EMA (50-Day)** - reacts more slowly to price changes and reflects the medium-to-longer-term trend rather than short-term swings

### Momentum Indicators
- **RSI (14-Day)** — flags overbought or oversold conditions
- **MACD** — tracks the relationship between two EMAs to catch shifts in momentum
- **MACD Signal Line** — a smoothed version of MACD used to confirm those shifts rather than react to every wiggle

### Volatility Indicators
- **ATR (14-Day)** — measures how much the price is swinging, regardless of direction
- **Bollinger Bands (Upper/Lower)** — built from a 20-day moving average plus/minus two standard deviations, they widen during volatile stretches and narrow when the market calms down

### Volume Indicator
- **On-Balance Volume (OBV)** — accumulates volume in the direction of price movement, used to confirm whether a price trend has real conviction behind it

### Return-Based Features
- **Daily Return** — the simple day-over-day percentage change
- **Log Return** — the log-transformed version, which behaves better statistically for financial time series and is standard in most quant modeling

### Lag Features
- **Lag 1 / Lag 2 / Lag 3 Close** — the closing prices from the previous 1, 2, and 3 days, giving the model direct short-term historical context rather than relying solely on rolling-window indicators

### News Sentiment Feature
- **Average Daily Sentiment** — headlines from NifSent50 and FindKG are cleaned, combined, and sorted chronologically from 2013 onward, then scored using VADER (a rule-based sentiment analysis tool tuned for short text like headlines and social media). Since multiple headlines can land on the same day, scores are averaged into a single daily value before merging into the main dataset by date. Days with no matching news default to a neutral (0) score, so the absence of news doesn't get mistaken for negative sentiment.

## Target Variable

The model predicts **tomorrow's closing price**, created by shifting the `Close` column back by one day. In other words, each row's features (today's prices, indicators, and sentiment) are paired with the label of what the closing price turns out to be the next trading day — that's the supervised learning signal the model learns from.

## Exploratory Data Analysis

A handful of visualizations help build intuition about the data before modelling:

- Closing price trend over time
- Trading volume trend
- Distribution of closing prices
- Correlation heatmap (useful for spotting redundant or highly collinear features)
- Boxplots for outlier detection
- Daily return distribution
- Actual vs. predicted closing prices (post-modelling)

## Machine Learning Models

Four models are trained and compared:

- **Linear Regression** — a simple baseline to measure whether the more complex models are actually adding value
- **Random Forest Regressor** — captures non-linear relationships between features without much tuning
- **XGBoost Regressor** — a gradient-boosted tree model, typically strong on tabular/structured data like this
- **LSTM (Long Short-Term Memory)** — a recurrent neural network suited to sequential data, tested here to see if it captures temporal patterns the tree-based models miss

## Model Evaluation

Each model's predictions are scored using:

- **MAE** (Mean Absolute Error) — average magnitude of error, in the same units as price
- **MSE** (Mean Squared Error) — penalises larger errors more heavily
- **RMSE** (Root Mean Squared Error) — same penalty behaviour as MSE, but back in interpretable price units
- **R² Score** — how much of the variance in closing price the model explains

## Tools and Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- yfinance
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- TA (Technical Analysis Library)
- VADER (vaderSentiment) — for news sentiment scoring
- kagglehub — for pulling the news datasets

## Future Improvements

- Using live, real-time stock and news data instead of static historical pulls
- Expanding sentiment analysis beyond VADER to a transformer-based model (e.g., FinBERT) for more nuanced financial-language understanding
- Fine-tuning model hyperparameters more rigorously (grid search / Bayesian optimisation)
- Deploying the model as a web application for live predictions

### Note

This project is intended for educational purposes, to demonstrate how machine learning — combined with technical indicators and news sentiment — can be applied to financial data analysis. Stock prices are influenced by countless unpredictable factors, and this model's predictions should not be treated as financial or investment advice.
