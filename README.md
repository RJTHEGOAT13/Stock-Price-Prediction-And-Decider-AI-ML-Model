# Stock-Price-Prediction-And-Decider-AI-ML-Model

# ML-Driven Stock Trading Strategy

**Automated sentiment‑driven trading** using Lumibot, FinBERT, and the Alpaca API. Backtest on historical data from Yahoo Finance before deploying live.

---

## 🚀 Features

- **Sentiment‑Driven Trades**  
  Uses FinBERT to classify financial news as **positive** or **negative** and makes buy/sell decisions.

- **Dynamic Position Sizing**  
  Calculates trade size based on available cash and a **user‑defined risk percentage**.

- **Bracket Orders**  
  Automatically sets **take‑profit** and **stop‑loss** levels for every trade.

- **Backtesting Support**  
  Simulates strategy performance over historical data to validate before going live.

---

## 🛠 Technology Stack

- **Lumibot** – Automated trading framework  
- **FinBERT** – NLP model for financial sentiment analysis  
- **Alpaca API** – Commission‑free trading platform  
- **Yahoo Finance** – Historical market data source  
- **Python 3.8+** with:
  - `transformers`
  - `torch`
  - `alpaca-trade-api`
  - `matplotlib`
  - `pandas`

---

## 📈 How It Works

1. **Fetch News**  
   Pulls the last 3 days of headlines for your chosen symbol via Alpaca’s news endpoint.

2. **Sentiment Analysis**  
   Applies **FinBERT** to assign a **probability** and **label** (`positive`, `negative`, `neutral`).

3. **Generate Trade Signal**  
   - **Buy** when sentiment is **positive** with confidence > 99.9%.  
   - **Sell** when sentiment is **negative** with confidence > 99.9%.

4. **Position Sizing & Risk Management**  
   - Computes how many shares to trade based on **`cash_at_risk`**.  
   - Places **bracket orders** with configurable take‑profit/stop‑loss levels.

5. **Backtesting**  
   - Runs over a specified date range using **YahooDataBacktesting**.  
   - Outputs performance metrics (total return, Sharpe, drawdown, win rate, etc.).

---

## ⚙️ Requirements

- **Python 3.8+**  
- Install dependencies:
  ```bash
  pip install lumibot transformers torch alpaca-trade-api matplotlib pandas


  # requirements.txt

# Core trading framework
lumibot

# Sentiment analysis
transformers
torch

# Broker API
alpaca-trade-api

# Data handling & plotting
pandas
matplotlib
