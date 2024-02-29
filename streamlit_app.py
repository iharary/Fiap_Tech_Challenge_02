import pandas as pd
import numpy as np 
import streamlit as st
from functions.calculation import (
    busca_cotacao_bolsa,  
    datas_faltantes, 
    decomposicao_sazonal,
    rolmean_rolstd, 
    teste_dickey_fuller,
    transformacao_locaritmica,
    media_movel,
    subtracao_MV_STL,
    calcula_derivada,
    wmape,
    renomeia_colunas,
    split_treino_teste,
    calcular_e_exibir_diferenca,
    hyperparametros_prophet,
    exibir_periodo_datas
)
from functions.plota_grafico import (
    plot_seasonal_decompose, 
    plot_cotacao_ibovespa, 
    plot_rolling_statistics, 
    plot_transformacao_logaritmica,
    plot_media_movel,
    plot_subtracao_MV_STL,
    plot_forecast,
    plot_prophet_1,
    plot_prophet_2,
    plot_prophet_3   
)
from models.modelos import naive_model, auto_arima_model, prophet_model
from PIL import Image
import sys
from prophet.diagnostics import cross_validation, performance_metrics

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


# Load the image
image = Image.open('image\Ibovespa.jpeg')

new_height = 200
image = image.resize((image.size[0], new_height))

# Display the image in Streamlit
st.image(image,  use_column_width=True)

#st.title('Ibovespa - Modelo Preditivo')
st.markdown("<h1 style='text-align: center; color: blue;'>Ibovespa - Modelo Preditivo</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Selecione as datas do Ibovespa')

st.markdown("<br>", unsafe_allow_html=True)

#tickers = ['^BVSP','PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 'WEGE3.SA', 'MGLU3.SA', 'VVAR3.SA', 'GNDI3.SA', 'NTCO3.SA']
tickers = ['^BVSP']

col1, col2, col3 = st.columns(3)
ticker = col1.selectbox('Selecione o ticker da ação:', tickers)
start_date = col2.date_input('Data inicial:', pd.to_datetime('2020-01-01'))
end_date = col3.date_input('Data final:', pd.to_datetime('2024-02-01'))

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Selecione parâmetros do modelo')

st.markdown("<br>", unsafe_allow_html=True)

col_treino, col_teste = st.columns(2)
dias_treino = col_treino.number_input ('Quantidade de dias de treino:', 1000)
dias_teste = col_teste.number_input('Quantidade de dias de teste:', 10)

if dias_treino == 0 or dias_teste == 0:
    st.warning('Por favor, selecione a quantidade de dias de treino e teste.')
    sys.exit()


st.markdown("<br>", unsafe_allow_html=True)

if st.button('Executar'):
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Busca Cotação Ibovespa
    df = busca_cotacao_bolsa(ticker, start_date, end_date)
    
    #quantidade de dias de treino e teste não pode ser maior que df
    qtd_dias_ibovespa = len(df)
    
    if dias_treino + dias_teste > qtd_dias_ibovespa:
        st.warning(f"A quantidade de dias de treino e teste não pode ser maior que {qtd_dias_ibovespa}. ")
        sys.exit()
    
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader('Cotação de Fechamento do Ibovespa')
        st.dataframe(df, width=400, height=300)
    
    
    with col2:
        st.subheader('Análise Gráfica do Índice IBOVESPA')
        #Plota o gráfico do Ibovespa    
        plot_cotacao_ibovespa(df)
    
    #Preenche datas faltantes com ultimo valor conhecido
    df_close_last = datas_faltantes(df)
    
    # Análise da Estacionariedade da Série Temporal
    st.markdown("<hr>", unsafe_allow_html=True)
    
    
    
    
    
    
    st.subheader('01. Análise da Estacionariedade da Série Temporal')
           
    #Decomposicao
    st.markdown('<h2 style="font-size: 1.5em;">Decomposição</h2>', unsafe_allow_html=True)
        
    decomposition = decomposicao_sazonal(df_close_last , 'additive',30)
    
    #Plotar Decomposição
    plot_seasonal_decompose(decomposition)

    st.markdown("<br>", unsafe_allow_html=True)
        
    #Rolling Statistics#
    st.markdown('<h2 style="font-size: 1.5em;">Análise de Estatísticas Móveis</h2>', unsafe_allow_html=True)
    
    rolmean, rolstd = rolmean_rolstd(df_close_last, 12)
    
     #Plotar Rolling Statistics
    plot_rolling_statistics(rolmean, rolstd)

    st.markdown('<h2 style="font-size: 1em;">Teste de Dickey-Fuller</h2>', unsafe_allow_html=True)

    retorno_teste_df = teste_dickey_fuller(df_close_last)    

    st.dataframe(retorno_teste_df, width=400, height=300)
    
    if retorno_teste_df['p-value'] > 0.05:
        st.error('A série NÃO é estacionária.')
    else:
        st.success('A série é estacionária.')
        
    st.markdown("<hr>", unsafe_allow_html=True)
        
    
    
    
    
    
    st.subheader('02. Conversão para Série Temporal Estacionária') 
    
    st.markdown('<h2 style="font-size: 1em;">Transformação Logarítmica aos Dados</h2>', unsafe_allow_html=True)

    st.markdown("""
     Vamos aplicar a transformação logarítmica para estabilizar a variância e aproximar a série de uma forma mais estacionária, 
     tornando-a mais adequada para análise e modelagem.
    """)  

    #Transformação Logaritica
    ts_log = transformacao_locaritmica(df_close_last)
    
    #Plota a transformação logaritmica
    plot_transformacao_logaritmica(ts_log)
    
    st.markdown('<h2 style="font-size: 1em;">Média Móvel</h2>', unsafe_allow_html=True)
    
    st.markdown("""
     Vamos aplicar a média móvel para suavizar a série e destacar tendências de longo prazo.
    """)  
    
    #Média Móvel
    moving_avg = media_movel(df_close_last, 12)
    
    #plota média móvel
    plot_media_movel(moving_avg, ts_log)
    
    #Subtração da Média Móvel da Série Temporal Logarítmica
    st.markdown('<h2 style="font-size: 1em;">Subtração da Média Móvel da Série Temporal Logarítmica</h2>', unsafe_allow_html=True)
    
    st.markdown("""
     Vamos remover as tendências e sazonalidades dos dados, tornando a série mais próxima de uma série estacionária..
    """)  
        
    ts_log_moving_avg_diff = subtracao_MV_STL(ts_log, moving_avg)
    
    rolmean, rolstd = rolmean_rolstd(ts_log_moving_avg_diff, 12)
    
    #Plota a subtração da média móvel da série temporal logarítmica
    plot_subtracao_MV_STL(ts_log_moving_avg_diff, rolmean, rolstd)
        
    retorno_teste_df = teste_dickey_fuller(ts_log_moving_avg_diff)    

    st.dataframe(retorno_teste_df, width=400, height=300)
    
    if retorno_teste_df['p-value'] > 0.05:
        st.error('A série NÃO é estacionária.')
    else:
        st.success('A série é estacionária.')
        df_diff = ts_log_moving_avg_diff.copy()
        
    #Fazer Derivada da Série Temporal    
    if retorno_teste_df['p-value'] > 0.05:
        
        df_diff = calcula_derivada(ts_log_moving_avg_diff)
        
        rolmean, rolstd = rolmean_rolstd(df_diff, 12)
        
        #Plota a subtração da média móvel da série temporal logarítmica
        plot_subtracao_MV_STL(df_diff, rolmean, rolstd)
        
        retorno_teste_df = teste_dickey_fuller(df_diff)
        
        if retorno_teste_df['p-value'] > 0.05:
            st.error('A série NÃO é estacionária.')
            sys.exit()
        else:
            st.success('A série é estacionária.')
            
    st.markdown("<hr>", unsafe_allow_html=True)        
            
    
    
    
    
    
    
    
    
    st.subheader('03. Abordagens de Modelagem')
    
    #renomear o campo date para ds, campo close para y e criar o campo unique_id
    df_modelo = renomeia_colunas(df_diff)
    
    # Dividindo os dados em treino e teste
    treino, teste, h =  split_treino_teste(df_modelo, dias_treino, dias_teste)
    
    ##Intervalo de Datas Utilizadas
    st.markdown('<h2 style="font-size: 1.5em;">Período das Datas Selecionadas</h2>', unsafe_allow_html=True)
    
    inicio_treino, fim_treino, inicio_teste, fim_teste =  exibir_periodo_datas(treino, teste)
    
    # Exibindo as informações no Streamlit
    st.markdown('<h2 style="font-size: 1.0em;">Conjunto de Treino</h2>', unsafe_allow_html=True)
    st.write(f"**Início:** {inicio_treino.strftime('%d/%m/%Y')}, **Fim:** {fim_treino.strftime('%d/%m/%Y')}")
    
    st.markdown('<h2 style="font-size: 1.0em;">Conjunto de Teste</h2>', unsafe_allow_html=True)
    st.write(f"**Início:** {inicio_teste.strftime('%d/%m/%Y')}, **Fim:** {fim_teste.strftime('%d/%m/%Y')}")
    
    
    
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    #Modelo Base Navi
    st.markdown('<h2 style="font-size: 1.5em;">Modelo Base Navi</h2>', unsafe_allow_html=True)
    
    model1, forecast_df1 = naive_model(treino, teste, h)
    
    wmape1 = wmape(forecast_df1['y'].values, forecast_df1['Naive'].values)
    
    st.metric(label="WMAPE", value=f"{wmape1:.2%}")

    if wmape1 < 0.05:
        st.success('O modelo é bom.')
    else:     
        st.error('O modelo é ruim.')
        
    #Plota o gráfico do forecast
    plot_forecast(model1, treino, forecast_df1)
    
    #AutoARIMA
    st.markdown('<h2 style="font-size: 1.5em;">AutoARIMA</h2>', unsafe_allow_html=True)
    
    model2, forecast_df2 = auto_arima_model(treino, teste, h)
    
    wmape2 = wmape(forecast_df2['y'].values, forecast_df2['AutoARIMA'].values)
    
    st.metric(label="WMAPE", value=f"{wmape2:.2%}")

    if wmape2 < 0.05:
        st.success('O modelo é bom.')
    else:     
        st.error('O modelo é ruim.')
        
    #Plota o gráfico do forecast
    plot_forecast(model2, treino, forecast_df2)
    
    # Para o Prophet, utilizar somente dados encontrados e sem a necessidade de transformação estacionária
    #renomear o campo date para ds, campo close para y e criar o campo unique_id
    df_modelo_prophet  = renomeia_colunas(df_close_last)
    
    # Dividindo os dados em treino e teste
    treino_prophet, teste_prophet, h =  split_treino_teste(df_modelo_prophet, dias_treino, dias_teste)
    
    #Prophet
    st.markdown('<h2 style="font-size: 1.5em;">Prophet</h2>', unsafe_allow_html=True)
    
    #model3, forecast3, forecast_df3 = prophet_model(treino_prophet, teste_prophet, h, 0.05, 0.05)
    
    #Hyperparameter Tuning
    with st.spinner('Aguarde... Calculando Hyperparameter Tuning.'):
        
        st.markdown('<h2 style="font-size: 1.0em;">Hyperparameter Tuning</h2>', unsafe_allow_html=True) 
        
        best_params, best_mae = hyperparametros_prophet(treino_prophet)
        
        st.write(f"Melhores hiperparâmetros: {best_params}")
        st.write(f"Melhor MAE: {best_mae}")
    
    
    model3, forecast3, forecast_df3 = prophet_model(treino_prophet, teste_prophet, h, best_params['changepoint_prior_scale'], best_params['seasonality_prior_scale'])
    
    wmape3 = wmape(forecast_df3['y'].values, forecast_df3['yhat'].values)
    
    st.metric(label="WMAPE", value=f"{wmape3:.2%}")

    if wmape3 < 0.05:
        st.success('O modelo é bom.')
    else:     
        st.error('O modelo é ruim.')
    
    st.markdown('<h2 style="font-size: 1.0em;">Evolução e Previsão Futura do Índice Ibovespa</h2>', unsafe_allow_html=True)   
    #Plota o gráfico do Prophet
    plot_prophet_1(model3, forecast3)
    
    st.markdown('<h2 style="font-size: 1.0em;">Componentes da Análise Temporal: Tendências e Sazonalidade do Índice Ibovespa</h2>', unsafe_allow_html=True)    
    #Plota o gráfico do Prophet
    plot_prophet_2(model3, forecast3)
 
    st.markdown('<h2 style="font-size: 1.0em;">Análise de Tendências do Índice Ibovespa: Comparação entre Dados de Treino, Reais e Previsões</h2>', unsafe_allow_html=True)   
    #Plota o  gráfico do Prophet
    plot_prophet_3(forecast_df3, treino_prophet)
    
    
    
    dif = calcular_e_exibir_diferenca(forecast_df3)
    
    # Exibe o DataFrame no Streamlit
    st.write("Diferença Percentual Absoluta entre Valores Reais e Previstos:")
    st.dataframe(dif)
    
    periodo_total = dias_treino + dias_teste
    
    #Cross-Validation
    #df_cv = cross_validation(model3)
     
    # Cross-validation
    with st.spinner('Aguarde... Processando.'):
        st.markdown('<h2 style="font-size: 1.5em;">Cross-validation</h2>', unsafe_allow_html=True) 
        df_cv = cross_validation(model3, initial='365 days', period='30 days', horizon='90 days')
            
        # Avaliação das métricas de desempenho
        df_p = performance_metrics(df_cv) 
                
        st.dataframe(df_p, width=800, height=300)
            
        st.success('Processamento concluído!')