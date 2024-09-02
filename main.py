import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from dotenv import load_dotenv
load_dotenv()

st.title('Welcome to your Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

if not ticker:
   st.subheader('Please input your stock ticker symbol in the sidebar and select the date range.') 
   st.subheader('For Indian stocks (NSE), use \'.NS\' after the ticker')
   st.text('Note: START DATE AND END DATE CANNOT BE THE SAME')

else:
    data = yf.download(ticker, start = start_date, end = end_date)
    figure = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)
    st.plotly_chart(figure)

    pricing_data, fundamental_data, news = st.tabs(['Pricing Data', 'Fundamental Data', 'News'])

    with pricing_data:
       st.header('Price Movements')
       data2 = data
       data2['% Change'] = data2['Adj Close'] / data2['Adj Close'].shift(1) - 1
       st.write(data2)
       annual_return = data2['% Change'].mean()*252*100
       st.write(f'Annual Return is {annual_return}%')
       stdev = np.std(data2['% Change']) * np.sqrt(252)
       st.write(f'Standard Deviation is {(stdev*100)}%')
       st.write(f'Risk Adj Return is {annual_return/(stdev*100)}')

    with fundamental_data:
       key = os.getenv('ALPHAVANTAGE_API_KEY')
       fd_api = FundamentalData(key, output_format='pandas')
       st.subheader('Balance Sheet')
       balance_sheet = fd_api.get_balance_sheet_annual(symbol=ticker)[0]
       bs = balance_sheet.T[2:]
       bs.columns = list(balance_sheet.T.iloc[0])
       st.write(bs)
       st.subheader('Income Statement')
       income_statement = fd_api.get_income_statement_annual(symbol=ticker)[0]
       is1 = income_statement.T[2:]
       is1.columns = list(income_statement.T.iloc[0])
       st.write(is1)
       st.subheader('Cash Flow Statement')
       cash_flow = fd_api.get_cash_flow_annual(symbol=ticker)[0]
       cf = cash_flow.T[2:]
       cf.columns = list(cash_flow.T.iloc[0])
       st.write(cf)


    with news:
       st.header(f'News of {ticker}')
       sn = StockNews(ticker, save_news=False)
       df_news = sn.read_rss()
       for i in range(5):
          st.subheader(f'News {i+1}')
          st.write(df_news['published'][i])   
          st.write(df_news['title'][i])
          st.write(df_news['summary'][i])
          title_sentiment = df_news['sentiment_title'][i]
          st.write(f'Title Sentiment: {title_sentiment}') 
          news_sentiment = df_news['sentiment_summary'][i]
          st.write(f'News Sentiment: {news_sentiment}')
         