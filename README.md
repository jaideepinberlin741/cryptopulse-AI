# CryptoPulse AI
[![Project Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://github.com/your-username/cryptopulse-ai)

### Short-Term Crypto Market Prediction with News Heatmaps

CryptoPulse AI is a **machine learning research project** that explores whether **short-term cryptocurrency price movements** can be predicted using **technical indicators and time-series models**.

The project combines **quantitative price analysis** with **crypto news context**, presenting insights through an interactive **Streamlit dashboard**.

---

# 📌 Project Motivation

Cryptocurrency markets are **highly volatile** and often influenced by both **technical signals** and **breaking news events**.

Most tools focus on either:

* Technical analysis (charts, indicators)
* News or sentiment analysis

Rarely do they combine both in a **transparent, research-oriented system**.

CryptoPulse AI aims to bridge this gap by:

* Predicting short-term price movement using **machine learning models**
* Providing **news heatmaps** to help interpret periods of volatility

---

# 🎯 Project Goals

The main objectives of this project are:

* Build a **time-series machine learning pipeline** for crypto price prediction
* Compare **traditional ML models** with **deep learning models**
* Implement **walk-forward validation** to avoid look-ahead bias
* Evaluate models using **backtesting simulations**
* Provide an **interactive dashboard** with charts, predictions, and news context

---

# 🏗 System Architecture

The system consists of the following components:

1. **Data Ingestion**

   * Crypto OHLC data from APIs
   * Crypto news headlines from news APIs

2. **Feature Engineering**

   * Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
   * Volatility and momentum features

3. **Machine Learning Pipeline**

   * Traditional ML models (XGBoost, LightGBM)
   * Deep learning models (LSTM / sequence models)

4. **Evaluation & Backtesting**

   * Walk-forward validation
   * Trading strategy simulation

5. **Visualization Dashboard**

   * OHLC price charts
   * Indicator visualizations
   * Model predictions
   * Backtesting results
   * News heatmap

---

# 📊 Example Workflow

```
Crypto API / News API
        │
        ▼
Data Collection
        │
        ▼
Feature Engineering
        │
        ▼
ML Model Training
        │
        ▼
Evaluation & Backtesting
        │
        ▼
Saved Models
        │
        ▼
Streamlit Dashboard
```

---
# CryptoPulse AI – System Architecture Diagram
                         +----------------------+
                         |   Crypto Data APIs   |
                         |  (Binance / CoinGecko)|
                         +----------+-----------+
                                    |
                                    |
                                    v
                          +-------------------+
                          |   OHLC Data       |
                          |   Ingestion       |
                          | (Python scripts)  |
                          +---------+---------+
                                    |
                                    v
                              +-----------+
                              | Raw Data  |
                              | Storage   |
                              | data/raw  |
                              +-----------+
                                    |
                                    |
                                    v
                         +---------------------+
                         | Feature Engineering |
                         |  Technical Indicators|
                         | (RSI, MACD, SMA...) |
                         +----------+----------+
                                    |
                                    v
                           +------------------+
                           | ML Dataset       |
                           | Builder          |
                           | (Supervised TS)  |
                           +--------+---------+
                                    |
                                    v
                    +--------------------------------+
                    |  Model Training Pipeline       |
                    |                                |
                    |  - XGBoost                     |
                    |  - LightGBM                    |
                    |  - LSTM / Deep Learning        |
                    +---------------+----------------+
                                    |
                                    v
                       +----------------------------+
                       | Walk-Forward Validation    |
                       | Time-Series Evaluation     |
                       +-------------+--------------+
                                     |
                                     v
                          +----------------------+
                          | Backtesting Engine   |
                          | Trading Simulation   |
                          +-----------+----------+
                                      |
                                      v
                         +--------------------------+
                         | Trained Models           |
                         | models/trained_models    |
                         +-----------+--------------+
                                     |
                                     |
                                     v
        ------------------------------------------------------------
                              DASHBOARD LAYER
        ------------------------------------------------------------

                 +-----------------------------------+
                 |        Streamlit Dashboard        |
                 +-----------------------------------+
                 |                                   |
                 | 1. Price Charts (OHLC)            |
                 | 2. Technical Indicators           |
                 | 3. Model Predictions              |
                 | 4. Backtesting Results            |
                 | 5. News Heatmap                   |
                 |                                   |
                 +---------------+-------------------+
                                 |
                                 v
                       +----------------------+
                       |     End User         |
                       | (Research / Demo)    |
                       +----------------------+


        ------------------------------------------------------------
                              NEWS PIPELINE
        ------------------------------------------------------------

      +--------------------+
      | Crypto News API    |
      | (NewsAPI / GDELT)  |
      +---------+----------+
                |
                v
      +---------------------+
      | News Processing     |
      | Asset detection     |
      | Sentiment tagging   |
      +---------+-----------+
                |
                v
      +----------------------+
      | News Aggregation     |
      | by asset & time      |
      +----------+-----------+
                 |
                 v
      +-----------------------+
      | News Heatmap Dataset  |
      +----------+------------+
                 |
                 v
         Streamlit Visualization
---

# 🧠 Machine Learning Models

The project compares several model types:

### Traditional ML

* XGBoost
* LightGBM
* Random Forest (baseline)

### Deep Learning

* LSTM (Long Short-Term Memory)
* Temporal sequence models

Models are trained using **time-series validation techniques** such as **walk-forward validation**.

---

# 📈 Backtesting Strategy

To evaluate model usefulness in practice, a simple **rule-based trading strategy** is used:

Example:

```
If model predicts price increase → LONG
Else → HOLD / FLAT
```

The strategy is compared against:

* Buy & Hold
* Random predictions
* Moving average strategy

Performance metrics include:

* Returns
* Maximum drawdown
* Sharpe ratio
* Stability over time

---

# 📰 News Heatmap

CryptoPulse AI also includes a **news heatmap visualization**.

Features:

* Aggregates crypto-related headlines
* Groups by **asset and time window**
* Optional **sentiment classification**
* Displays **news intensity during volatility**

Example visualization:

```
Asset | Time Window | News Intensity
BTC   | 2025-01-01  | █████
ETH   | 2025-01-01  | ███
SOL   | 2025-01-01  | ██
```

This helps users **understand the context behind price movements**.

---

# 🖥 Dashboard

The **Streamlit dashboard** provides an interactive interface.

Features:

* OHLC price charts
* Technical indicators
* Model prediction plots
* Backtesting performance
* Crypto news heatmap

---

# 🛠 Tech Stack

### Core

* Python
* NumPy
* Pandas

### Machine Learning

* scikit-learn
* XGBoost
* LightGBM
* TensorFlow / PyTorch

### Technical Indicators

* pandas-ta
* TA-Lib

### Visualization

* Matplotlib
* Plotly
* mplfinance
* Streamlit

### Deployment

* Docker

---

# 📂 Project Structure

```
cryptopulse-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── news/
│
├── dashboard/
│
├── models/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ▶️ Running the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/cryptopulse-ai.git
cd cryptopulse-ai
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the dashboard

```
streamlit run dashboard/app.py
```

---

# 🐳 Docker Deployment

Build the container:

```
docker build -t cryptopulse-ai .
```

Run the container:

```
docker run -p 8501:8501 cryptopulse-ai
```

---

# ⚠️ Disclaimer

This project is **for research and educational purposes only**.

The models and trading simulations presented here **do not constitute financial advice**.

Cryptocurrency markets involve **significant risk**, and results from historical data **do not guarantee future performance**.

---

# 👨‍💻 Author

**Jaideep Naik**
**Jolanda Tinge**

Software Engineer | Machine Learning Enthusiast

---

# ⭐ Future Improvements

Possible extensions include:

* Transformer-based time-series models
* Advanced sentiment analysis for news
* Real-time crypto market streaming
* Portfolio optimization
* Automated trading integration

---

If you found this project useful, consider giving it a ⭐ on GitHub!
