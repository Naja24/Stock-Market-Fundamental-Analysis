# 📊 Stock Market Fundamental Analysis — Streamlit App

**Complete Financial Analysis System → Fundamental + Technical + Sentiment**

🔴 **Live Deployed App:** 👉 https://naja24-stock-market-fundamental-analysis-app-x1yal1.streamlit.app/

> This is the main research project combining:  
> ✔ Fundamental Analysis  
> ✔ Technical Indicators  
> ✔ Sentiment NLP  
> ✔ Weighted Scoring Model  
> ✔ Streamlit Dashboard

---

## 🚀 Overview

This project builds a **full-stack stock evaluation engine** using:

### 📘 1. Fundamental Analysis (80% weight)
- Valuation ratios (P/E, P/B, PEG, EV/EBITDA)
- Profitability (ROE, ROA, margins)
- Financial Health (Debt/Equity, Cash vs Debt)
- Earnings & Dividend profile
- Financial statements (Income, Balance Sheet, Cash Flow)

### 📙 2. Technical Analysis (10% weight)
- RSI
- SMA50 / EMA20
- Trend validation

### 📗 3. Sentiment Analysis (10% weight)
- News polarity
- Keyword scoring
- Market mood quantification

---

## 📂 Repository Structure
```
Stock Market Fundamental Analysis/
│
├── app.py
├── requirements.txt
└── src/
    ├── fundamental.py
    ├── technical.py
    ├── sentiment.py
    └── __init__.py
```

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Yahoo Finance API**
- **News API** (RapidAPI)
- **Vader Sentiment NLP**
- **Plotly**
- **Pandas**

---

## 🔧 Installation & Setup

### 1. Clone the Repo
```bash
git clone https://github.com/Naja24/Stock-Market-Fundamental-Analysis.git
cd Stock-Market-Fundamental-Analysis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

The app launches at: 👉 **http://localhost:8501**

---

## 🧠 System Architecture
```
Input Layer:
  - Yahoo Finance API
  - News API
  - Historical Data
         ↓
Processing Layer:
  - Fundamental Engine
  - Technical Engine
  - Sentiment Engine
         ↓
Logic Layer:
  - Weighted Scoring (80-10-10)
         ↓
Output Layer:
  - Streamlit UI
  - Charts & Metrics
  - Final Recommendation
```

---

## ✨ Features

### 🔵 Fundamental Analysis
- Ratios
- Profitability
- Liquidity
- Cash Flow
- Statement Trend Charts

### 🔵 Technical Indicators
- RSI
- EMA20
- SMA50
- Short-term trend detection

### 🔵 Sentiment Engine
- Polarity score
- News summarization
- Contrarian signals

### 🔵 Final Recommendation
- **Buy / Hold / Sell**
- Based on weighted score

---

## 📸 Screenshots

*(Add after deployment)*

- Homepage
- Ratio dashboard
- Statements
- Sentiment chart
- Final verdict

---

## 🔗 Related Repo (Technical Models)

For **deep learning price forecasting** with LSTM, N-BEATS, and TFT:  
👉 **[Stock-Market-Analysis](https://github.com/Naja24/Stock-Market-Analysis)**

---

## 📜 License

MIT License.

---

## 🙏 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/Naja24/Stock-Market-Fundamental-Analysis/issues).

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a ⭐!

---

## 📧 Contact

For questions or collaborations, feel free to reach out via GitHub issues.

---

## 🚀 Try It Now

**Live App:** https://naja24-stock-market-fundamental-analysis-app-x1yal1.streamlit.app/

Analyze any stock instantly with comprehensive fundamental, technical, and sentiment analysis!

> **⚠️ Important Note:**  
> The technical analysis component currently uses Linear Regression for trend prediction and should be considered supplementary to the fundamental analysis. For more robust time-series forecasting, please refer to our advanced deep learning models (LSTM, N-BEATS, TFT) in the [companion repository](https://github.com/Naja24/Stock-Market-Analysis). An enhanced technical analysis module with improved predictive capabilities is under development and will be released in future updates.
