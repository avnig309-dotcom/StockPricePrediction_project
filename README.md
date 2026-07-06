**Stock Price Prediction Using Machine Learning**

**About the Project**

The stock market is influenced by many factors, making it difficult to predict future prices accurately. In this project, we use Machine Learning to predict the next day's closing price of the NIFTY 50 index using historical stock market data.

Instead of relying only on the original stock prices, we generate several technical indicators such as Moving Averages, RSI, MACD, Bollinger Bands, and ATR. These indicators help the model understand market trends, momentum, and volatility more effectively.

The project follows a complete machine learning workflow, starting from collecting the data and preparing it for analysis, to training multiple regression models and comparing their performance.


**Project Objectives**

Predict the next day's closing price of the NIFTY 50 index.

Analyze historical stock market data.

Create technical indicators to improve prediction accuracy.

Explore trends and patterns through data visualization.

Train and compare different regression models.

Evaluate model performance using standard regression metrics.


**Dataset**

The historical stock market data is downloaded from Yahoo Finance using the yfinance library.

Ticker: ^NSEI (NIFTY 50)

The dataset contains the following columns:

Date

Open

High

Low

Close

Adjusted Close

Volume

The data spans multiple years, allowing the model to learn long-term market behaviour.


**Data Preprocessing**

Before training the model, the dataset is checked for:

Missing values
Duplicate records
Correct data types
Chronological order of dates

After the data is verified, additional technical indicators are created and any rows containing missing values generated during feature engineering are removed.

**Feature Engineering**

To improve the model's performance, several technical indicators are added to the dataset.

**Trend Indicators**

SMA (20-Day)

SMA (50-Day)

EMA (20-Day)

**Momentum Indicators**

RSI (14-Day)

MACD

MACD Signal Line

**Volatility Indicators**

ATR (14-Day)

Bollinger Upper Band

Bollinger Lower Band

**Volume Indicator**

On Balance Volume (OBV)

**Return-Based Features**

Daily Return

Log Return

These indicators provide more information about the market than the raw price values alone and help the model identify important patterns.

**Target Variable**

The model predicts the next day's closing price.

The target column is created by shifting the closing price one day ahead, allowing the model to learn from today's market data to estimate tomorrow's closing price.

**Exploratory Data Analysis**

Several visualizations are created to better understand the dataset, including:

Closing Price Trend

Trading Volume Trend

Distribution of Closing Prices

Correlation Heatmap

Boxplots for Outlier Detection

Daily Return Distribution

Actual vs Predicted Closing Prices

These graphs help identify trends, relationships between variables, and unusual observations in the data.

**Machine Learning Models**

The following regression models are implemented and compared:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Long Short-Term Memory (LSTM)

Linear Regression serves as a simple baseline model, while the other models are used to capture more complex relationships within the data.

**Model Evaluation**

The performance of each model is evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

These metrics help compare how accurately each model predicts the next day's closing price.

**Tools and Libraries**
Python

Pandas

NumPy

Matplotlib

Seaborn

yfinance

Scikit-learn

XGBoost

TensorFlow / Keras

TA (Technical Analysis Library)

**Future Improvements**

Some possible improvements to this project include:

Using live stock market data for real-time predictions.

Including news sentiment and economic indicators.

Fine-tuning model parameters for better performance.

Deploying the model as a web application.

**Note:** 

This project is intended for educational purposes to demonstrate the application of machine learning in financial data analysis. Since stock prices are influenced by many unpredictable factors, the model's predictions should not be considered financial or investment advice.
