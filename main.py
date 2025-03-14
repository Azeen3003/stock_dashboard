import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from stocknews import StockNews
from dotenv import load_dotenv
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Load environment variables
load_dotenv()

st.title('Welcome to your Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
period = st.sidebar.selectbox('Select Time Period', ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'])

if not ticker.strip():
    st.error("Ticker cannot be empty. Please enter a valid stock symbol.")
else:
    try:
       data = yf.Ticker("TSLA").history(period="6mo")
       if data.empty:
          raise ValueError("Yahoo Finance returned empty data.")
    except Exception as e:
          print(f"Error fetching data: {e}")

    
    figure = px.line(data, x=data.index, y=data['Close'].squeeze(), title=f"{ticker} Stock Price")
    st.plotly_chart(figure)
    
    pricing_data, fundamental_data, news, prediction = st.tabs(['Pricing Data', 'Fundamental Data', 'News', 'Prediction'])
    
    with pricing_data:
        st.header('Price Movements')
        data['% Change'] = data['Close'].pct_change() * 100
        st.write(data)
        
        annual_return = data['% Change'].mean() * 252
        stdev = np.std(data['% Change']) * np.sqrt(252)
        st.write(f'Annual Return: {annual_return:.2f}%')
        st.write(f'Standard Deviation: {stdev:.2f}%')
        st.write(f'Risk-Adjusted Return: {annual_return/stdev:.2f}')
    
    with fundamental_data:
        st.subheader('Fundamental Data')
        try:
            info = stock.info
            if not info:
                st.warning("No fundamental data available.")
            else:
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
                st.write(f"**Revenue:** {info.get('totalRevenue', 'N/A')}")
                st.write(f"**Net Income:** {info.get('netIncomeToCommon', 'N/A')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
        except Exception as e:
            st.error(f"Error fetching fundamental data: {e}")
    
    with news:
        st.header(f'News for {ticker}')
        try:
            sn = StockNews(ticker, save_news=False)
            df_news = sn.read_rss()
            if df_news.empty:
                st.warning("No news articles found for this ticker.")
            else:
                for i in range(min(5, len(df_news))):
                    st.subheader(f'News {i+1}')
                    st.write(df_news['published'][i])
                    st.write(df_news['title'][i])
                    st.write(df_news['summary'][i])
                    st.write(f"Title Sentiment: {df_news['sentiment_title'][i]}")
                    st.write(f"News Sentiment: {df_news['sentiment_summary'][i]}")
        except Exception as e:
            st.error(f"News data unavailable: {e}")
    
    with prediction:
        st.header(f'Prediction for {ticker}')
        n_years = st.slider('Years of prediction:', 1, 4)
        period_days = n_years * 365
        
        df_train = data[['Close']].reset_index()
        df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        
        try:
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period_days)
            forecast = m.predict(future)
            
            st.subheader('Forecast Data')
            st.write(forecast.tail())
            
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
        except Exception as e:
            st.error(f"Prediction model error: {e}")
