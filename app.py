import pandas as pd
import numpy as np 
import streamlit as st
from functions.busca_cotacao_bolsa import busca_cotacao_bolsa
#from functions.decomposicao_sazonal import decomposicao_sazonal
from functions.plota_grafico import plot_seasonal_decompose

tickers = ['^BVSP','PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 'WEGE3.SA', 'MGLU3.SA', 'VVAR3.SA', 'GNDI3.SA', 'NTCO3.SA']


st.title('Cotação de Ações')

ticker = st.selectbox('Selecione o ticker da ação:', tickers)

#ticker = st.text_input('Digite o ticker da ação:', 'PETR4.SA')

start_date = st.date_input('Data inicial:', pd.to_datetime('2023-01-01'))

end_date = st.date_input('Data final:', pd.to_datetime('2024-01-01'))

if st.button('Buscar'):
    df = busca_cotacao_bolsa(ticker, start_date, end_date)
    st.dataframe(df)
    st.line_chart(df['Close'])  # Plot the closing price of the stock
    
    #Plota a decomposição sazonal
    plot_seasonal_decompose(df, 'Close', model='additive', period=30)   
