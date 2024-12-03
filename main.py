import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as date
import plotly.express as px
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from dotenv import load_dotenv
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
load_dotenv()

st.title('Welcome to your Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

if not ticker.strip():
    st.error("Ticker cannot be empty. Please enter a valid stock symbol.")

elif start_date >= end_date:
    st.error("Start date must be earlier than end date.")

else:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data available for this ticker or date range.")
    except Exception as e:
        st.error("Failed to retrieve data. Please try again later.")

    figure = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)
    st.plotly_chart(figure)

    pricing_data, fundamental_data, news, prediction = st.tabs(['Pricing Data', 'Fundamental Data', 'News', 'Prediction'])

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
       
       try:
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

       except ValueError as e:
           if "API call frequency" in str(e):
               st.error("API call limit reached. Please try again later or upgrade your API plan.")
           else:
               st.error(f"An error occurred: {str(e)}")
       


    with news:
       st.header(f'News of {ticker}')
       try:
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

       except Exception as e:
           if "HTTP" in str(e):
               st.error("News data unavailable due to API limit. Please try again later.")
           else:
               st.error(f"An error occurred: {str(e)}")
       

   
    with prediction:
       st.header(f'Prediction of {ticker}')
       START = '2014-12-03'
       END = end_date

       n_years = st.slider('Years of prediction:', 1, 4)
       period = n_years * 365

       @st.cache_data
       def load_data(ticker):
         data = yf.download(ticker, START, END)
         data.reset_index(inplace=True)
         return data
       
       data_load_state = st.text('Loading data...')
       data = load_data(ticker)
       data_load_state.text('Loading data... done!')

       st.subheader('Raw data')
       st.write(data.tail())

       # Plot raw data
       def plot_raw_data():
          fig = go.Figure()
          fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
          fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
          fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
          st.plotly_chart(fig)

       plot_raw_data()

       # Predict forecast with Prophet.
       df_train = data[['Date','Close']]
       df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

       m = Prophet()
       m.fit(df_train)
       future = m.make_future_dataframe(periods=period)
       forecast = m.predict(future)

       # Show and plot forecast
       st.subheader('Forecast data')
       st.write(forecast.tail())
           
       st.write(f'Forecast plot for {n_years} years')
       fig1 = plot_plotly(m, forecast)
       st.plotly_chart(fig1)
       
       st.write("Forecast components")
       fig2 = m.plot_components(forecast)
       st.write(fig2)
    