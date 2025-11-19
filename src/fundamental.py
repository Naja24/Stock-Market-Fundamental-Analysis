# import streamlit as st
# import yfinance as yf
# import pandas as pd

# def display_fundamental_analysis(ticker_symbol):
#     st.header(f"Fundamental Analysis: {ticker_symbol}")
    
#     stock = yf.Ticker(ticker_symbol)
#     info = stock.info

#     # 1. Company Overview
#     st.subheader("1. Company Overview")
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         # Try to display logo if available
#         if 'logo_url' in info and info['logo_url']:
#             st.image(info['logo_url'])
#     with col2:
#         st.write(info.get('longBusinessSummary', 'No summary available.'))
    
#     # 2. Key Metrics
#     st.subheader("2. Key Financial Ratios")
#     metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
#     with metrics_col1:
#         st.metric("Market Cap", f"₹ {info.get('marketCap', 0) / 10**7:,.2f} Cr")
#         st.metric("Current Price", f"₹ {info.get('currentPrice', 0)}")
    
#     with metrics_col2:
#         st.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
#         st.metric("Book Value", info.get('bookValue', 'N/A'))
    
#     with metrics_col3:
#         st.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
#         st.metric("Debt to Equity", info.get('debtToEquity', 'N/A'))
        
#     with metrics_col4:
#         st.metric("Dividend Yield", f"{info.get('dividendRate', 0)*100:.2f}%")
#         st.metric("52 Week High", info.get('fiftyTwoWeekHigh', 'N/A'))

#     # 3. Financial Statements
#     st.subheader("3. Financial Statements")
#     tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    
#     with tab1:
#         st.write("### Annual Income Statement")
#         st.dataframe(stock.financials)
    
#     with tab2:
#         st.write("### Annual Balance Sheet")
#         st.dataframe(stock.balance_sheet)
    
#     with tab3:
#         st.write("### Annual Cash Flow")
#         st.dataframe(stock.cashflow)

#     # 4. Shareholding Pattern (if available)
#     st.subheader("4. Institutional Holders")
#     try:
#         sh = stock.institutional_holders
#         if sh is not None and not sh.empty:
#             st.dataframe(sh)
#         else:
#             st.info("Institutional holding data not directly available via API.")
#     except:
#         st.info("Could not fetch shareholding pattern.")


import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def format_large_number(num):
    """Format large numbers in Indian format (Lakhs/Crores)"""
    if num is None or pd.isna(num):
        return "N/A"
    
    num = float(num)
    if abs(num) >= 1e7:  # Crores
        return f"₹ {num/1e7:,.2f} Cr"
    elif abs(num) >= 1e5:  # Lakhs
        return f"₹ {num/1e5:,.2f} L"
    elif abs(num) >= 1e3:  # Thousands
        return f"₹ {num/1e3:,.2f} K"
    else:
        return f"₹ {num:,.2f}"

def get_valuation_status(pe_ratio, industry_avg_pe=20):
    """Determine if stock is overvalued or undervalued"""
    if pe_ratio is None or pd.isna(pe_ratio):
        return "N/A", "gray"
    
    if pe_ratio < 0:
        return "Negative Earnings", "red"
    elif pe_ratio < 15:
        return "Undervalued", "green"
    elif pe_ratio < 25:
        return "Fair Value", "orange"
    else:
        return "Overvalued", "red"

def display_fundamental_analysis(ticker_symbol):
    st.header(f"🏢 Fundamental Analysis: {ticker_symbol}")
    
    with st.spinner('Fetching comprehensive company data...'):
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

    if not info or len(info) < 5:
        st.error("❌ Unable to fetch fundamental data for this ticker. Please verify the symbol.")
        return

    # ========== SECTION 1: COMPANY OVERVIEW ==========
    st.subheader("📋 Company Overview")
    
    col_logo, col_info = st.columns([1, 4])
    
    with col_logo:
        # Display logo if available
        logo_url = info.get('logo_url') or info.get('logoUrl')
        if logo_url:
            try:
                st.image(logo_url, width=150)
            except:
                st.info("📊")
        else:
            st.info("📊")
        
        # Company classification
        st.markdown("#### Classification")
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        country = info.get('country', 'India')
        
        st.write(f"**Sector:** {sector}")
        st.write(f"**Industry:** {industry}")
        st.write(f"**Country:** {country}")
        
        # Exchange info
        exchange = info.get('exchange', 'N/A')
        st.write(f"**Exchange:** {exchange}")
    
    with col_info:
        company_name = info.get('longName') or info.get('shortName', ticker_symbol)
        st.markdown(f"### {company_name}")
        
        # Business Summary
        summary = info.get('longBusinessSummary', 'No business summary available.')
        with st.expander("📖 Read Business Summary", expanded=False):
            st.write(summary)
        
        # Quick Stats Row
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            employees = info.get('fullTimeEmployees')
            if employees:
                st.metric("👥 Employees", f"{employees:,}")
            else:
                st.metric("👥 Employees", "N/A")
        
        with stat_col2:
            founded = info.get('founded', 'N/A')
            st.metric("📅 Founded", founded)
        
        with stat_col3:
            website = info.get('website', '#')
            if website and website != '#':
                st.markdown(f"🌐 **Website**")
                st.markdown(f"[Visit Site]({website})")
            else:
                st.metric("🌐 Website", "N/A")
        
        with stat_col4:
            phone = info.get('phone', 'N/A')
            if phone and phone != 'N/A':
                st.metric("📞 Phone", phone)
            else:
                st.metric("📞 Contact", "N/A")

    st.divider()

    # ========== SECTION 2: KEY FINANCIAL METRICS ==========
    st.subheader("💰 Key Financial Ratios & Valuation Metrics")
    
    # Row 1: Price & Market Cap Metrics
    st.markdown("#### 💵 Market Metrics")
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    
    with mcol1:
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('previousClose')
        
        if current_price and prev_close:
            change = ((current_price - prev_close) / prev_close * 100)
            st.metric("Current Price", f"₹ {current_price:.2f}", f"{change:+.2f}%")
        elif current_price:
            st.metric("Current Price", f"₹ {current_price:.2f}")
        else:
            st.metric("Current Price", "N/A")
    
    with mcol2:
        market_cap = info.get('marketCap')
        if market_cap:
            market_cap_cr = market_cap / 1e7
            st.metric("Market Cap", f"₹ {market_cap_cr:,.0f} Cr")
        else:
            st.metric("Market Cap", "N/A")
    
    with mcol3:
        volume = info.get('volume') or info.get('regularMarketVolume')
        avg_volume = info.get('averageVolume')
        
        if volume:
            st.metric("Volume", f"{volume:,}")
            if avg_volume:
                vol_ratio = (volume / avg_volume - 1) * 100
                st.caption(f"Avg: {avg_volume:,} ({vol_ratio:+.1f}%)")
        else:
            st.metric("Volume", "N/A")
    
    with mcol4:
        week_52_high = info.get('fiftyTwoWeekHigh')
        week_52_low = info.get('fiftyTwoWeekLow')
        
        if week_52_high:
            st.metric("52W High", f"₹ {week_52_high:.2f}")
        else:
            st.metric("52W High", "N/A")
        
        if week_52_low:
            st.caption(f"52W Low: ₹ {week_52_low:.2f}")
    
    with mcol5:
        beta = info.get('beta')
        if beta:
            st.metric("Beta (Volatility)", f"{beta:.2f}")
            if beta > 1.5:
                st.caption("🔴 High Volatility")
            elif beta < 0.8:
                st.caption("🟢 Low Volatility")
            else:
                st.caption("⚪ Moderate")
        else:
            st.metric("Beta", "N/A")

    st.markdown("")
    
    # Row 2: Valuation Ratios
    st.markdown("#### 📊 Valuation Ratios")
    vcol1, vcol2, vcol3, vcol4, vcol5 = st.columns(5)
    
    with vcol1:
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        if pe_ratio:
            valuation_status, status_color = get_valuation_status(pe_ratio)
            st.metric("P/E Ratio", f"{pe_ratio:.2f}")
            
            if status_color == "green":
                st.success(f"✅ {valuation_status}")
            elif status_color == "red":
                st.error(f"⚠️ {valuation_status}")
            else:
                st.info(f"ℹ️ {valuation_status}")
        else:
            st.metric("P/E Ratio", "N/A")
    
    with vcol2:
        pb_ratio = info.get('priceToBook')
        if pb_ratio:
            st.metric("P/B Ratio", f"{pb_ratio:.2f}")
            if pb_ratio < 1:
                st.caption("🟢 Below Book Value")
            elif pb_ratio > 3:
                st.caption("🔴 Premium Valuation")
        else:
            st.metric("P/B Ratio", "N/A")
    
    with vcol3:
        ps_ratio = info.get('priceToSalesTrailing12Months')
        if ps_ratio:
            st.metric("P/S Ratio", f"{ps_ratio:.2f}")
        else:
            st.metric("P/S Ratio", "N/A")
    
    with vcol4:
        peg_ratio = info.get('pegRatio')
        if peg_ratio:
            st.metric("PEG Ratio", f"{peg_ratio:.2f}")
            if peg_ratio < 1:
                st.caption("🟢 Good Value")
            elif peg_ratio > 2:
                st.caption("🔴 Expensive")
        else:
            st.metric("PEG Ratio", "N/A")
    
    with vcol5:
        ev_ebitda = info.get('enterpriseToEbitda')
        if ev_ebitda:
            st.metric("EV/EBITDA", f"{ev_ebitda:.2f}")
        else:
            st.metric("EV/EBITDA", "N/A")

    st.markdown("")
    
    # Row 3: Profitability & Efficiency
    st.markdown("#### 📈 Profitability & Efficiency Metrics")
    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
    
    with pcol1:
        roe = info.get('returnOnEquity')
        if roe:
            st.metric("ROE", f"{roe*100:.2f}%")
            if roe > 0.20:
                st.caption("🟢 Excellent (>20%)")
            elif roe > 0.15:
                st.caption("🟢 Very Good (>15%)")
            elif roe > 0.10:
                st.caption("⚪ Good (>10%)")
            else:
                st.caption("🔴 Weak (<10%)")
        else:
            st.metric("ROE", "N/A")
    
    with pcol2:
        roa = info.get('returnOnAssets')
        if roa:
            st.metric("ROA", f"{roa*100:.2f}%")
        else:
            st.metric("ROA", "N/A")
    
    with pcol3:
        profit_margin = info.get('profitMargins')
        if profit_margin:
            st.metric("Profit Margin", f"{profit_margin*100:.2f}%")
            if profit_margin > 0.20:
                st.caption("🟢 High Margin")
            elif profit_margin > 0.10:
                st.caption("⚪ Moderate")
            else:
                st.caption("🔴 Low Margin")
        else:
            st.metric("Profit Margin", "N/A")
    
    with pcol4:
        operating_margin = info.get('operatingMargins')
        if operating_margin:
            st.metric("Operating Margin", f"{operating_margin*100:.2f}%")
        else:
            st.metric("Operating Margin", "N/A")
    
    with pcol5:
        gross_margin = info.get('grossMargins')
        if gross_margin:
            st.metric("Gross Margin", f"{gross_margin*100:.2f}%")
        else:
            st.metric("Gross Margin", "N/A")

    st.markdown("")
    
    # Row 4: Financial Health & Leverage
    st.markdown("#### 🏦 Financial Health & Leverage")
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
    
    with fcol1:
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity:
            st.metric("Debt/Equity", f"{debt_to_equity:.2f}")
            if debt_to_equity < 0.5:
                st.caption("🟢 Low Leverage")
            elif debt_to_equity < 1.0:
                st.caption("⚪ Moderate")
            else:
                st.caption("🔴 High Leverage")
        else:
            st.metric("Debt/Equity", "N/A")
    
    with fcol2:
        current_ratio = info.get('currentRatio')
        if current_ratio:
            st.metric("Current Ratio", f"{current_ratio:.2f}")
            if current_ratio > 2:
                st.caption("🟢 Strong Liquidity")
            elif current_ratio > 1:
                st.caption("⚪ Adequate")
            else:
                st.caption("🔴 Weak Liquidity")
        else:
            st.metric("Current Ratio", "N/A")
    
    with fcol3:
        quick_ratio = info.get('quickRatio')
        if quick_ratio:
            st.metric("Quick Ratio", f"{quick_ratio:.2f}")
        else:
            st.metric("Quick Ratio", "N/A")
    
    with fcol4:
        total_cash = info.get('totalCash')
        if total_cash:
            st.metric("Total Cash", format_large_number(total_cash))
        else:
            st.metric("Total Cash", "N/A")
    
    with fcol5:
        free_cashflow = info.get('freeCashflow')
        if free_cashflow:
            st.metric("Free Cash Flow", format_large_number(free_cashflow))
        else:
            st.metric("Free Cash Flow", "N/A")

    st.markdown("")
    
    # Row 5: Earnings & Dividends
    st.markdown("#### 💸 Earnings & Dividend Information")
    ecol1, ecol2, ecol3, ecol4, ecol5 = st.columns(5)
    
    with ecol1:
        eps = info.get('trailingEps')
        if eps:
            st.metric("EPS (TTM)", f"₹ {eps:.2f}")
        else:
            st.metric("EPS (TTM)", "N/A")
    
    with ecol2:
        forward_eps = info.get('forwardEps')
        if forward_eps:
            st.metric("Forward EPS", f"₹ {forward_eps:.2f}")
        else:
            st.metric("Forward EPS", "N/A")
    
    with ecol3:
        book_value = info.get('bookValue')
        if book_value:
            st.metric("Book Value", f"₹ {book_value:.2f}")
        else:
            st.metric("Book Value", "N/A")
    
    with ecol4:
        dividend_yield = info.get('dividendYield')
        dividend_rate = info.get('dividendRate')
        
        if dividend_yield:
            st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
            if dividend_rate:
                st.caption(f"Rate: ₹ {dividend_rate:.2f}")
        else:
            st.metric("Dividend Yield", "N/A")
    
    with ecol5:
        payout_ratio = info.get('payoutRatio')
        if payout_ratio:
            st.metric("Payout Ratio", f"{payout_ratio*100:.2f}%")
            if payout_ratio > 0.8:
                st.caption("⚠️ High Payout")
            elif payout_ratio > 0.5:
                st.caption("⚪ Moderate")
            else:
                st.caption("🟢 Conservative")
        else:
            st.metric("Payout Ratio", "N/A")

    st.divider()

    # ========== SECTION 3: FINANCIAL STATEMENTS ==========
    st.subheader("📊 Financial Statements Analysis")
    
    tab1, tab2, tab3 = st.tabs(["💵 Income Statement", "🏦 Balance Sheet", "💸 Cash Flow"])
    
    with tab1:
        st.markdown("### Annual Income Statement")
        try:
            financials = stock.financials
            if financials is not None and not financials.empty:
                # Display full dataframe
                st.dataframe(
                    financials.style.format("{:,.0f}"),
                    use_container_width=True,
                    height=400
                )
                
                # Revenue & Profit Visualization
                st.markdown("#### 📈 Revenue and Profit Trends")
                
                fig_income = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Revenue Trend', 'Net Income Trend')
                )
                
                # Revenue chart
                if 'Total Revenue' in financials.index:
                    revenue_data = financials.loc['Total Revenue'].sort_index()
                    fig_income.add_trace(
                        go.Bar(
                            x=revenue_data.index.astype(str),
                            y=revenue_data.values,
                            marker_color='#1976d2',
                            name='Revenue'
                        ),
                        row=1, col=1
                    )
                
                # Net Income chart
                if 'Net Income' in financials.index:
                    income_data = financials.loc['Net Income'].sort_index()
                    fig_income.add_trace(
                        go.Bar(
                            x=income_data.index.astype(str),
                            y=income_data.values,
                            marker_color='#4caf50',
                            name='Net Income'
                        ),
                        row=1, col=2
                    )
                
                fig_income.update_layout(
                    height=400,
                    showlegend=False,
                    template='plotly_white'
                )
                fig_income.update_xaxes(title_text="Year", row=1, col=1)
                fig_income.update_xaxes(title_text="Year", row=1, col=2)
                fig_income.update_yaxes(title_text="Amount (₹)", row=1, col=1)
                fig_income.update_yaxes(title_text="Amount (₹)", row=1, col=2)
                
                st.plotly_chart(fig_income, use_container_width=True)
                
                # Growth rates
                if 'Total Revenue' in financials.index and len(financials.columns) >= 2:
                    latest_revenue = financials.loc['Total Revenue'].iloc[0]
                    prev_revenue = financials.loc['Total Revenue'].iloc[1]
                    revenue_growth = ((latest_revenue - prev_revenue) / prev_revenue * 100)
                    
                    growth_col1, growth_col2 = st.columns(2)
                    with growth_col1:
                        st.metric("Revenue Growth (YoY)", f"{revenue_growth:+.2f}%")
                    
                    if 'Net Income' in financials.index:
                        latest_income = financials.loc['Net Income'].iloc[0]
                        prev_income = financials.loc['Net Income'].iloc[1]
                        if prev_income != 0:
                            income_growth = ((latest_income - prev_income) / prev_income * 100)
                            with growth_col2:
                                st.metric("Net Income Growth (YoY)", f"{income_growth:+.2f}%")
            else:
                st.info("📊 Financial statement data not available for this stock")
        except Exception as e:
            st.error(f"❌ Could not fetch income statement: {str(e)}")
    
    with tab2:
        st.markdown("### Annual Balance Sheet")
        try:
            balance_sheet = stock.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                st.dataframe(
                    balance_sheet.style.format("{:,.0f}"),
                    use_container_width=True,
                    height=400
                )
                
                # Assets vs Liabilities Visualization
                st.markdown("#### ⚖️ Assets vs Liabilities")
                
                if 'Total Assets' in balance_sheet.index and 'Total Liabilities Net Minority Interest' in balance_sheet.index:
                    assets = balance_sheet.loc['Total Assets'].sort_index()
                    liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].sort_index()
                    
                    fig_balance = go.Figure()
                    fig_balance.add_trace(go.Bar(
                        x=assets.index.astype(str),
                        y=assets.values,
                        name='Total Assets',
                        marker_color='#4caf50'
                    ))
                    fig_balance.add_trace(go.Bar(
                        x=liabilities.index.astype(str),
                        y=liabilities.values,
                        name='Total Liabilities',
                        marker_color='#f44336'
                    ))
                    
                    fig_balance.update_layout(
                        title='Assets vs Liabilities Over Time',
                        xaxis_title='Year',
                        yaxis_title='Amount (₹)',
                        height=400,
                        template='plotly_white',
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig_balance, use_container_width=True)
            else:
                st.info("📊 Balance sheet data not available for this stock")
        except Exception as e:
            st.error(f"❌ Could not fetch balance sheet: {str(e)}")
    
    with tab3:
        st.markdown("### Annual Cash Flow Statement")
        try:
            cashflow = stock.cashflow
            if cashflow is not None and not cashflow.empty:
                st.dataframe(
                    cashflow.style.format("{:,.0f}"),
                    use_container_width=True,
                    height=400
                )
                
                # Cash Flow Visualization
                st.markdown("#### 💰 Cash Flow Trends")
                
                fig_cf = go.Figure()
                
                cash_flow_items = {
                    'Operating Cash Flow': '#4caf50',
                    'Investing Cash Flow': '#ff9800',
                    'Financing Cash Flow': '#2196f3'
                }
                
                for item, color in cash_flow_items.items():
                    if item in cashflow.index:
                        data = cashflow.loc[item].sort_index()
                        fig_cf.add_trace(go.Scatter(
                            x=data.index.astype(str),
                            y=data.values,
                            mode='lines+markers',
                            name=item,
                            line=dict(width=3, color=color),
                            marker=dict(size=10)
                        ))
                
                fig_cf.update_layout(
                    title='Cash Flow Components',
                    xaxis_title='Year',
                    yaxis_title='Amount (₹)',
                    height=400,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_cf, use_container_width=True)
                
                # Free Cash Flow margin
                if 'Operating Cash Flow' in cashflow.index and 'Total Revenue' in stock.financials.index:
                    ocf_col, fcf_col = st.columns(2)
                    
                    with ocf_col:
                        latest_ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
                        st.metric("Latest Operating Cash Flow", format_large_number(latest_ocf))
                    
                    if 'Free Cash Flow' in cashflow.index:
                        with fcf_col:
                            latest_fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                            st.metric("Latest Free Cash Flow", format_large_number(latest_fcf))
            else:
                st.info("📊 Cash flow data not available for this stock")
        except Exception as e:
            st.error(f"❌ Could not fetch cash flow: {str(e)}")

    st.divider()

    # ========== SECTION 4: SHAREHOLDING PATTERN ==========
    st.subheader("👥 Ownership & Shareholding")
    
    scol1, scol2 = st.columns(2)
    
    with scol1:
        st.markdown("#### Major Holders Summary")
        try:
            major_holders = stock.major_holders
            if major_holders is not None and not major_holders.empty:
                st.dataframe(major_holders, use_container_width=True, hide_index=True)
            else:
                st.info("Major holders data not available")
        except:
            st.info("Could not fetch major holders information")
    
    with scol2:
        st.markdown("#### Insider Ownership")
        insider_pct = info.get('heldPercentInsiders')
        institution_pct = info.get('heldPercentInstitutions')
        
        if insider_pct is not None or institution_pct is not None:
            ownership_data = []
            if insider_pct:
                ownership_data.append({'Category': 'Insiders', 'Percentage': insider_pct * 100})
            if institution_pct:
                ownership_data.append({'Category': 'Institutions', 'Percentage': institution_pct * 100})
            
            if ownership_data:
                remaining = 100 - sum([d['Percentage'] for d in ownership_data])
                if remaining > 0:
                    ownership_data.append({'Category': 'Public/Others', 'Percentage': remaining})
                
                fig_ownership = go.Figure(data=[go.Pie(
                    labels=[d['Category'] for d in ownership_data],
                    values=[d['Percentage'] for d in ownership_data],
                    hole=0.4,
                    marker_colors=['#1976d2', '#4caf50', '#ff9800']
                )])
                
                fig_ownership.update_layout(
                    title='Ownership Distribution',
                    height=300
                )
                
                st.plotly_chart(fig_ownership, use_container_width=True)
        else:
            st.info("Ownership distribution data not available")
    
    # Institutional Holders
    st.markdown("#### 🏛️ Top Institutional Holders")
    try:
        institutional = stock.institutional_holders
        if institutional is not None and not institutional.empty:
            st.dataframe(institutional, use_container_width=True, hide_index=True)
        else:
            st.info("Institutional holding details not available")
    except:
        st.info("Could not fetch institutional holders data")

    st.divider()

    # ========== SECTION 5: ANALYST RECOMMENDATIONS ==========
    st.subheader("📈 Analyst Recommendations & Price Targets")
    
    acol1, acol2 = st.columns([2, 1])
    
    with acol1:
        st.markdown("#### Recent Recommendations")
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                recent_recs = recommendations.tail(15)
                st.dataframe(recent_recs, use_container_width=True)
            else:
                st.info("No analyst recommendations available")
        except:
            st.info("Could not fetch analyst recommendations")
    
    with acol2:
        st.markdown("#### Price Targets")
        
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        target_mean = info.get('targetMeanPrice')
        target_median = info.get('targetMedianPrice')
        
        if target_mean:
            st.metric("Mean Target", f"₹ {target_mean:.2f}")
            
            if current_price:
                upside = ((target_mean - current_price) / current_price * 100)
                st.metric("Potential Upside", f"{upside:+.2f}%")
        
        if target_high and target_low:
            st.write(f"**Range:** ₹ {target_low:.2f} - ₹ {target_high:.2f}")
        
        if target_median:
            st.write(f"**Median:** ₹ {target_median:.2f}")
        
        # Recommendation summary
        num_analysts = info.get('numberOfAnalystOpinions')
        if num_analysts:
            st.write(f"**Analysts Covering:** {num_analysts}")
        
        recommendation_key = info.get('recommendationKey')
        if recommendation_key:
            rec_display = recommendation_key.upper()
            if 'buy' in recommendation_key.lower():
                st.success(f"**Consensus:** {rec_display}")
            elif 'sell' in recommendation_key.lower():
                st.error(f"**Consensus:** {rec_display}")
            else:
                st.info(f"**Consensus:** {rec_display}")

    st.divider()

    # Final Note
    st.info("💡 **Note:** All financial data is sourced from Yahoo Finance. Please verify critical information from official sources before making investment decisions.")