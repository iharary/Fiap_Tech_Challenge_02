import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

def busca_cotacao(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        return df
    except:
        return pd.DataFrame()
    
        