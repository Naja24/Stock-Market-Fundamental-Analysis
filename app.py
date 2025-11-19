import streamlit as st
from src.fundamental import display_fundamental_analysis
from src.technical import display_technical_analysis
from src.sentiment import display_sentiment_analysis

# Page Config
st.set_page_config(page_title="Indian Stock Analysis", layout="wide")

# Sidebar
st.sidebar.title("🇮🇳 Stock Analysis Tool")
st.sidebar.write("Enter the NSE stock symbol below:")
# Default to TCS (Tata Consultancy Services)
ticker = st.sidebar.text_input("Ticker Symbol", "TCS.NS").upper()

if not ticker.endswith(".NS"):
    st.sidebar.warning("Note: For Indian stocks, please add '.NS' (e.g., RELIANCE.NS)")

st.sidebar.markdown("---")
st.sidebar.info("This project performs Fundamental, Technical, and Sentiment analysis for BTP.")
st.sidebar.markdown("[GitHub Repo (Technical Focus)](https://github.com/Naja24/Stock-Market-Analysis)")

# Main Title
st.title(f"📊 Stock Market Analysis: {ticker}")

# Tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📈 Technical & Prediction", "🏢 Fundamental Analysis", "📰 Sentiment Analysis"])

with tab1:
    # Technical Analysis Section
    try:
        display_technical_analysis(ticker)
    except Exception as e:
        st.error(f"Error in Technical Analysis: {e}")

with tab2:
    # Fundamental Analysis Section
    try:
        display_fundamental_analysis(ticker)
    except Exception as e:
        st.error(f"Error in Fundamental Analysis: {e}")

with tab3:
    # Sentiment Analysis Section
    try:
        display_sentiment_analysis(ticker)
    except Exception as e:
        st.error(f"Error in Sentiment Analysis: {e}")