import pandas as pd
import numpy as np 
import streamlit as st
from functions.busca_cotacao_bolsa import busca_cotacao_bolsa
#from functions.decomposicao_sazonal import decomposicao_sazonal
from functions.plota_grafico import plot_seasonal_decompose

st.set_page_config(
    page_title='Ibovespa - Modelo Preditivo', 
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'https://www.google.com.br/',
        'Report a bug': "https://www.google.com.br/",
        'About': "Esse app foi desenvolvido pelo Grupo 38."
     }
)

tickers = ['^BVSP','PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 'WEGE3.SA', 'MGLU3.SA', 'VVAR3.SA', 'GNDI3.SA', 'NTCO3.SA']
models = ['Naive','SeasonalNaive','SeasonalWindowAverage','AutoARIMA']

col1, col2, col3, col4 = st.columns(4)
ticker = col1.selectbox('Selecione o ticker da ação:', tickers)
start_date = col2.date_input('Data inicial:', pd.to_datetime('2023-01-01'))
end_date = col3.date_input('Data final:', pd.to_datetime('2024-01-01'))
model = col4.selectbox('Selecione o modelo:', models)

with col1:
    st.text('Fazer decomposição sazonal?')
    
decompisicao = col1.checkbox('Decomposição Sazonal?')

if st.button('Executar'):
    df = busca_cotacao_bolsa(ticker, start_date, end_date)
    st.dataframe(df)
    st.line_chart(df['Close'])  # Plot the closing price of the stock
    
    #Plota a decomposição sazonal
    if decompisicao:
        plot_seasonal_decompose(df, 'Close', model='additive', period=30)   
