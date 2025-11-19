# import streamlit as st
# import yfinance as yf
# from textblob import TextBlob
# import pandas as pd

# def display_sentiment_analysis(ticker_symbol):
#     st.header(f"Sentiment Driven Analysis: {ticker_symbol}")
    
#     stock = yf.Ticker(ticker_symbol)
    
#     # Fetch News
#     try:
#         news_list = stock.news
#     except:
#         st.error("Could not fetch live news.")
#         return

#     if not news_list:
#         st.warning("No recent news found for this stock.")
#         return

#     st.subheader("Recent News & Sentiment Score")
    
#     sentiment_scores = []
    
#     for item in news_list:
#         title = item.get('title', '')
#         publisher = item.get('publisher', 'Unknown')
#         link = item.get('link', '#')
        
#         # Perform Sentiment Analysis
#         blob = TextBlob(title)
#         polarity = blob.sentiment.polarity  # -1 to 1
#         sentiment_scores.append(polarity)
        
#         # Determine Label
#         if polarity > 0.1:
#             emoji = "🟢 Bullish"
#             color = "green"
#         elif polarity < -0.1:
#             emoji = "🔴 Bearish"
#             color = "red"
#         else:
#             emoji = "⚪ Neutral"
#             color = "gray"

#         # Display News Card
#         with st.expander(f"{emoji} {title} ({publisher})"):
#             st.write(f"**Sentiment Score:** {polarity:.2f}")
#             st.markdown(f"[Read Article]({link})")

#     # Aggregate Sentiment
#     avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
#     st.divider()
#     st.subheader("Overall Market Sentiment")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
    
#     with col2:
#         if avg_sentiment > 0.05:
#             st.success("The overall news sentiment is POSITIVE.")
#         elif avg_sentiment < -0.05:
#             st.error("The overall news sentiment is NEGATIVE.")
#         else:
#             st.info("The overall news sentiment is NEUTRAL.")


import streamlit as st
import yfinance as yf
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

def get_sentiment_label(polarity):
    """Convert polarity score to label with detailed classification"""
    if polarity > 0.25:
        return "🟢 Very Bullish", "#2e7d32", 3, "strong_buy"
    elif polarity > 0.10:
        return "🟢 Bullish", "#66bb6a", 2, "buy"
    elif polarity > 0.03:
        return "🟡 Slightly Bullish", "#fdd835", 1, "weak_buy"
    elif polarity < -0.25:
        return "🔴 Very Bearish", "#c62828", -3, "strong_sell"
    elif polarity < -0.10:
        return "🔴 Bearish", "#ef5350", -2, "sell"
    elif polarity < -0.03:
        return "🟠 Slightly Bearish", "#ff9800", -1, "weak_sell"
    else:
        return "⚪ Neutral", "#9e9e9e", 0, "hold"

def analyze_text_sentiment(text):
    """Enhanced sentiment analysis with keyword weighting"""
    # TextBlob base sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    text_lower = text.lower()
    
    # Strong bullish keywords (weight: 0.15)
    strong_bullish = [
        'surge', 'soar', 'rally', 'breakout', 'boom', 'skyrocket',
        'outstanding', 'exceptional', 'record high', 'all-time high',
        'massive gain', 'explosive growth'
    ]
    
    # Moderate bullish keywords (weight: 0.10)
    moderate_bullish = [
        'profit', 'gain', 'growth', 'rise', 'increase', 'positive',
        'beat', 'exceed', 'strong', 'bullish', 'upgrade', 'outperform',
        'buy', 'success', 'improve', 'up', 'higher', 'advance',
        'win', 'expanding', 'momentum', 'optimistic'
    ]
    
    # Weak bullish keywords (weight: 0.05)
    weak_bullish = [
        'stable', 'steady', 'maintain', 'hold', 'support',
        'resilient', 'potential', 'opportunity'
    ]
    
    # Strong bearish keywords (weight: -0.15)
    strong_bearish = [
        'crash', 'plunge', 'collapse', 'disaster', 'crisis',
        'massive loss', 'bankruptcy', 'default', 'scandal'
    ]
    
    # Moderate bearish keywords (weight: -0.10)
    moderate_bearish = [
        'fall', 'drop', 'loss', 'decline', 'decrease', 'negative',
        'miss', 'weak', 'bearish', 'downgrade', 'sell', 'underperform',
        'concern', 'risk', 'warning', 'down', 'lower', 'slump',
        'fail', 'struggle', 'disappointing'
    ]
    
    # Weak bearish keywords (weight: -0.05)
    weak_bearish = [
        'caution', 'uncertainty', 'volatile', 'pressure',
        'challenge', 'slow', 'softness'
    ]
    
    # Calculate keyword sentiment
    keyword_score = 0
    
    # Count strong bullish
    for word in strong_bullish:
        if word in text_lower:
            keyword_score += 0.15
    
    # Count moderate bullish
    for word in moderate_bullish:
        if word in text_lower:
            keyword_score += 0.10
    
    # Count weak bullish
    for word in weak_bullish:
        if word in text_lower:
            keyword_score += 0.05
    
    # Count strong bearish
    for word in strong_bearish:
        if word in text_lower:
            keyword_score -= 0.15
    
    # Count moderate bearish
    for word in moderate_bearish:
        if word in text_lower:
            keyword_score -= 0.10
    
    # Count weak bearish
    for word in weak_bearish:
        if word in text_lower:
            keyword_score -= 0.05
    
    # Check for percentage changes in text
    percent_pattern = r'([+-]?\d+(?:\.\d+)?)\s*%'
    percentages = re.findall(percent_pattern, text)
    
    for pct_str in percentages:
        try:
            pct = float(pct_str)
            if pct > 5:
                keyword_score += 0.08
            elif pct > 2:
                keyword_score += 0.05
            elif pct < -5:
                keyword_score -= 0.08
            elif pct < -2:
                keyword_score -= 0.05
        except:
            pass
    
    # Combine TextBlob sentiment with keyword analysis (60-40 split)
    final_polarity = (0.6 * polarity) + (0.4 * keyword_score)
    
    # Clamp between -1 and 1
    final_polarity = max(-1, min(1, final_polarity))
    
    return final_polarity

def calculate_sentiment_confidence(scores):
    """Calculate confidence level based on score consistency"""
    if not scores:
        return 0
    
    import numpy as np
    std_dev = np.std(scores)
    mean_abs = np.mean([abs(s) for s in scores])
    
    # Lower std_dev and higher mean_abs = higher confidence
    if std_dev < 0.2 and mean_abs > 0.3:
        return 90
    elif std_dev < 0.3 and mean_abs > 0.2:
        return 75
    elif std_dev < 0.4 and mean_abs > 0.1:
        return 60
    else:
        return 45

def display_sentiment_analysis(ticker_symbol):
    st.header(f"📰 AI-Powered Sentiment Analysis: {ticker_symbol}")
    
    with st.spinner('🔍 Analyzing market sentiment from news sources...'):
        stock = yf.Ticker(ticker_symbol)
        
        # Fetch News
        try:
            news_list = stock.news
        except Exception as e:
            st.error(f"❌ Could not fetch news data: {str(e)}")
            st.info("💡 This may be due to API limitations or network issues. Please try again later.")
            return

    if not news_list or len(news_list) == 0:
        st.warning("⚠️ No recent news articles found for this stock.")
        st.info("""
        **Possible reasons:**
        - The ticker symbol may not have recent news coverage
        - The stock may be less actively traded
        - Temporary API limitations
        
        **Suggestion:** Try a more actively traded stock like TCS.NS, RELIANCE.NS, or INFY.NS
        """)
        return

    st.success(f"✅ Found **{len(news_list)}** recent news articles")
    
    # Process all news articles
    sentiment_scores = []
    sentiment_categories = []
    news_data = []
    
    for idx, item in enumerate(news_list[:20]):  # Process up to 20 articles
        title = item.get('title', '')
        publisher = item.get('publisher', 'Unknown')
        link = item.get('link', '#')
        
        # Get timestamp
        timestamp = item.get('providerPublishTime')
        if timestamp:
            pub_date = datetime.fromtimestamp(timestamp)
            pub_date_str = pub_date.strftime('%Y-%m-%d %H:%M')
            days_ago = (datetime.now() - pub_date).days
        else:
            pub_date = None
            pub_date_str = 'N/A'
            days_ago = 999
        
        # Perform Enhanced Sentiment Analysis
        polarity = analyze_text_sentiment(title)
        sentiment_scores.append(polarity)
        
        # Get detailed label
        emoji_label, color, category, signal = get_sentiment_label(polarity)
        sentiment_categories.append(category)
        
        news_data.append({
            'title': title,
            'publisher': publisher,
            'link': link,
            'date': pub_date,
            'date_str': pub_date_str,
            'days_ago': days_ago,
            'polarity': polarity,
            'label': emoji_label,
            'color': color,
            'signal': signal
        })

    # Sort by date (most recent first)
    news_data = sorted(news_data, key=lambda x: x['days_ago'])

    # ========== SENTIMENT OVERVIEW DASHBOARD ==========
    st.divider()
    st.subheader("📊 Sentiment Overview Dashboard")
    
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    confidence = calculate_sentiment_confidence(sentiment_scores)
    
    # Count categories
    very_bullish = sentiment_categories.count(3)
    bullish = sentiment_categories.count(2)
    slightly_bullish = sentiment_categories.count(1)
    neutral = sentiment_categories.count(0)
    slightly_bearish = sentiment_categories.count(-1)
    bearish = sentiment_categories.count(-2)
    very_bearish = sentiment_categories.count(-3)
    
    total_positive = very_bullish + bullish + slightly_bullish
    total_negative = very_bearish + bearish + slightly_bearish
    
    # Key Metrics Row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "Average Sentiment",
            f"{avg_sentiment:.3f}",
            help="Range: -1 (Very Bearish) to +1 (Very Bullish)"
        )
        
        # Sentiment gauge
        if avg_sentiment > 0.15:
            st.success("🟢 **POSITIVE**")
        elif avg_sentiment < -0.15:
            st.error("🔴 **NEGATIVE**")
        else:
            st.info("⚪ **NEUTRAL**")
    
    with metric_col2:
        st.metric(
            "Confidence Level",
            f"{confidence}%",
            help="Based on sentiment consistency across articles"
        )
        
        if confidence > 75:
            st.success("High Confidence")
        elif confidence > 60:
            st.info("Moderate Confidence")
        else:
            st.warning("Low Confidence")
    
    with metric_col3:
        positive_pct = (total_positive / len(sentiment_scores)) * 100
        st.metric("Positive News", f"{positive_pct:.1f}%")
        st.progress(positive_pct / 100)
    
    with metric_col4:
        negative_pct = (total_negative / len(sentiment_scores)) * 100
        st.metric("Negative News", f"{negative_pct:.1f}%")
        st.progress(negative_pct / 100)

    # ========== TRADING SIGNAL ==========
    st.divider()
    st.subheader("🎯 AI Trading Signal")
    
    signal_col1, signal_col2 = st.columns([2, 1])
    
    with signal_col1:
        # Determine overall signal
        if avg_sentiment > 0.20 and confidence > 70:
            st.success("### 🟢 STRONG BUY SIGNAL")
            st.write("""
            **Analysis:** The overwhelming majority of news is positive with high confidence.
            - Positive sentiment dominates the news cycle
            - High confidence in bullish trend
            - Market momentum is strong
            """)
        elif avg_sentiment > 0.10 and confidence > 60:
            st.success("### 🟢 BUY SIGNAL")
            st.write("""
            **Analysis:** Positive sentiment with good confidence level.
            - More positive news than negative
            - Moderate to high confidence
            - Favorable market sentiment
            """)
        elif avg_sentiment > 0.03:
            st.info("### 🟡 WEAK BUY / HOLD")
            st.write("""
            **Analysis:** Slightly positive sentiment but mixed signals.
            - Marginal positive sentiment
            - Consider holding current positions
            - Monitor for stronger signals
            """)
        elif avg_sentiment < -0.20 and confidence > 70:
            st.error("### 🔴 STRONG SELL SIGNAL")
            st.write("""
            **Analysis:** Predominantly negative news with high confidence.
            - Bearish sentiment dominates
            - High confidence in negative trend
            - Consider protective measures
            """)
        elif avg_sentiment < -0.10 and confidence > 60:
            st.error("### 🔴 SELL SIGNAL")
            st.write("""
            **Analysis:** Negative sentiment with good confidence.
            - More negative than positive news
            - Moderate to high confidence
            - Caution advised
            """)
        elif avg_sentiment < -0.03:
            st.warning("### 🟠 WEAK SELL / HOLD")
            st.write("""
            **Analysis:** Slightly negative sentiment but uncertain.
            - Marginal negative sentiment
            - Hold and reassess
            - Watch for trend changes
            """)
        else:
            st.info("### ⚪ NEUTRAL / HOLD")
            st.write("""
            **Analysis:** Mixed sentiment with no clear direction.
            - Equal balance of positive and negative news
            - No strong trend detected
            - Maintain current positions and monitor
            """)
    
    with signal_col2:
        # Sentiment breakdown
        st.markdown("### 📈 Breakdown")
        st.write(f"🟢 Very Bullish: **{very_bullish}**")
        st.write(f"🟢 Bullish: **{bullish}**")
        st.write(f"🟡 Slightly Bullish: **{slightly_bullish}**")
        st.write(f"⚪ Neutral: **{neutral}**")
        st.write(f"🟠 Slightly Bearish: **{slightly_bearish}**")
        st.write(f"🔴 Bearish: **{bearish}**")
        st.write(f"🔴 Very Bearish: **{very_bearish}**")

    # ========== VISUALIZATIONS ==========
    st.divider()
    st.subheader("📊 Sentiment Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Pie Chart - Sentiment Distribution
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Very Bullish', 'Bullish', 'Slightly Bullish', 'Neutral', 
                    'Slightly Bearish', 'Bearish', 'Very Bearish'],
            values=[very_bullish, bullish, slightly_bullish, neutral,
                    slightly_bearish, bearish, very_bearish],
            hole=0.4,
            marker_colors=['#2e7d32', '#66bb6a', '#fdd835', '#9e9e9e',
                          '#ff9800', '#ef5350', '#c62828'],
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig_pie.update_layout(
            title='Sentiment Distribution',
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with viz_col2:
        # Gauge Chart - Overall Sentiment
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_sentiment,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Sentiment Score"},
            delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-1, -0.3], 'color': '#ffcdd2'},
                    {'range': [-0.3, -0.1], 'color': '#fff9c4'},
                    {'range': [-0.1, 0.1], 'color': '#f5f5f5'},
                    {'range': [0.1, 0.3], 'color': '#dcedc8'},
                    {'range': [0.3, 1], 'color': '#c8e6c9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig_gauge.update_layout(height=400, font={'size': 16})
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ========== SENTIMENT TIMELINE ==========
    st.divider()
    st.subheader("📈 Sentiment Timeline & Trends")
    
    # Create DataFrame
    news_df = pd.DataFrame(news_data)
    
    if not news_df.empty and news_df['date'].notna().any():
        # Filter out rows with no date
        news_df_dated = news_df[news_df['date'].notna()].copy()
        
        if not news_df_dated.empty:
            # Sort by date
            news_df_dated = news_df_dated.sort_values('date')
            
            # Create timeline chart
            fig_timeline = go.Figure()
            
            # Add scatter plot with color-coded points
            fig_timeline.add_trace(go.Scatter(
                x=news_df_dated['date'],
                y=news_df_dated['polarity'],
                mode='markers+lines',
                marker=dict(
                    size=12,
                    color=news_df_dated['polarity'],
                    colorscale=[
                        [0, '#c62828'],      # Very negative
                        [0.2, '#ef5350'],    # Negative
                        [0.4, '#ff9800'],    # Slightly negative
                        [0.5, '#9e9e9e'],    # Neutral
                        [0.6, '#fdd835'],    # Slightly positive
                        [0.8, '#66bb6a'],    # Positive
                        [1, '#2e7d32']       # Very positive
                    ],
                    showscale=True,
                    colorbar=dict(
                        title="Sentiment",
                        tickvals=[-0.75, -0.25, 0, 0.25, 0.75],
                        ticktext=['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish']
                    ),
                    line=dict(width=2, color='white')
                ),
                line=dict(width=2, color='rgba(100, 100, 100, 0.3)'),
                name='Sentiment',
                text=news_df_dated['title'],
                hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.3f}<br>Date: %{x}<extra></extra>'
            ))
            
            # Add reference lines
            fig_timeline.add_hline(y=0, line_dash="solid", line_color="gray", 
                                  annotation_text="Neutral", line_width=2)
            fig_timeline.add_hline(y=0.20, line_dash="dot", line_color="green", 
                                  annotation_text="Bullish Threshold", line_width=1)
            fig_timeline.add_hline(y=-0.20, line_dash="dot", line_color="red", 
                                  annotation_text="Bearish Threshold", line_width=1)
            
            # Add trend line
            if len(news_df_dated) > 2:
                from sklearn.linear_model import LinearRegression
                import numpy as np
                
                X = np.array(range(len(news_df_dated))).reshape(-1, 1)
                y = news_df_dated['polarity'].values
                
                lr = LinearRegression()
                lr.fit(X, y)
                trend = lr.predict(X)
                
                fig_timeline.add_trace(go.Scatter(
                    x=news_df_dated['date'],
                    y=trend,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='blue', width=3, dash='dash'),
                    hoverinfo='skip'
                ))
            
            fig_timeline.update_layout(
                title='Sentiment Over Time',
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                height=450,
                template='plotly_white',
                hovermode='closest',
                yaxis=dict(range=[-1, 1])
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Recent trend analysis
            if len(news_df_dated) >= 5:
                recent_5 = news_df_dated.tail(5)['polarity'].mean()
                overall_avg = news_df_dated['polarity'].mean()
                
                trend_col1, trend_col2 = st.columns(2)
                with trend_col1:
                    st.metric("Recent Trend (Last 5 Articles)", f"{recent_5:.3f}")
                with trend_col2:
                    trend_change = recent_5 - overall_avg
                    st.metric("Trend vs Overall", f"{trend_change:+.3f}",
                             "Improving" if trend_change > 0 else "Declining")

    # ========== NEWS ARTICLES DETAILS ==========
    st.divider()
    st.subheader(f"📰 Detailed News Analysis ({len(news_data)} Articles)")
    
    # Filter options
    filter_col1, filter_col2 = st.columns([1, 3])
    
    with filter_col1:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment:",
            ["All", "Bullish Only", "Bearish Only", "Neutral Only"]
        )
    
    # Apply filter
    if sentiment_filter == "Bullish Only":
        filtered_news = [n for n in news_data if n['polarity'] > 0.03]
    elif sentiment_filter == "Bearish Only":
        filtered_news = [n for n in news_data if n['polarity'] < -0.03]
    elif sentiment_filter == "Neutral Only":
        filtered_news = [n for n in news_data if -0.03 <= n['polarity'] <= 0.03]
    else:
        filtered_news = news_data
    
    st.write(f"Showing **{len(filtered_news)}** articles")
    
    # Display news cards
    for idx, item in enumerate(filtered_news[:15]):  # Show top 15
        with st.expander(
            f"{item['label']} | {item['title'][:100]}{'...' if len(item['title']) > 100 else ''} ({item['days_ago']} days ago)",
            expanded=(idx < 3)  # Expand first 3
        ):
            news_col1, news_col2 = st.columns([3, 1])
            
            with news_col1:
                st.markdown(f"**📰 {item['title']}**")
                st.write(f"**Publisher:** {item['publisher']}")
                st.write(f"**Published:** {item['date_str']}")
                
                # Show link
                if item['link'] and item['link'] != '#':
                    st.markdown(f"[🔗 Read Full Article]({item['link']})")
                
                # Interpretation
                if item['polarity'] > 0.2:
                    st.success("💡 **Interpretation:** Strong positive news that could drive prices up.")
                elif item['polarity'] > 0.1:
                    st.success("💡 **Interpretation:** Positive development, favorable for investors.")
                elif item['polarity'] < -0.2:
                    st.error("💡 **Interpretation:** Strong negative news that could pressure prices.")
                elif item['polarity'] < -0.1:
                    st.error("💡 **Interpretation:** Negative development, caution advised.")
                else:
                    st.info("💡 **Interpretation:** Neutral news, limited market impact expected.")
            
            with news_col2:
                # Sentiment score with color
                st.markdown(f"**Sentiment Score**")
                st.markdown(f"<h2 style='color: {item['color']};'>{item['polarity']:.3f}</h2>", 
                           unsafe_allow_html=True)
                
                # Visual bar
                normalized_score = (item['polarity'] + 1) / 2  # Convert -1:1 to 0:1
                st.progress(normalized_score)
                
                # Signal
                st.markdown(f"**Signal:** {item['signal'].upper().replace('_', ' ')}")

    # ========== DISCLAIMER ==========
    st.divider()
    st.info("""
    ### 💡 Important Notes on Sentiment Analysis
    
    **How It Works:**
    - Analyzes news headlines using Natural Language Processing (NLP)
    - Combines TextBlob sentiment with financial keyword analysis
    - Weights keywords based on their market impact significance
    
    **Limitations:**
    - Based on headlines only, not full article content
    - Cannot detect sarcasm or complex context
    - News sentiment is one factor among many in price movements
    - Past news sentiment doesn't guarantee future price direction
    
    **Best Practices:**
    - Use sentiment as a supplementary indicator, not the sole decision factor
    - Combine with technical and fundamental analysis
    - Consider the confidence level when interpreting signals
    - Always verify important news from primary sources
    
    ⚠️ **Disclaimer:** This analysis is for informational purposes only. Not financial advice.
    """)