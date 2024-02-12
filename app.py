import pandas as pd
import numpy as np 
import streamlit as st
from functions.busca_cotacao_bolsa import busca_cotacao


st.title('Cotação de Ações')

ticker = st.text_input('Digite o ticker da ação:', 'PETR4.SA')

start_date = st.date_input('Data inicial:', pd.to_datetime('2020-01-01'))

end_date = st.date_input('Data final:', pd.to_datetime('2021-01-01'))

if st.button('Buscar'):
    df = busca_cotacao(ticker, start_date, end_date)
    st.dataframe(df)
    st.line_chart(df['Close'])  # Plot the closing price of the stock                               

# Run the app with the following command:
