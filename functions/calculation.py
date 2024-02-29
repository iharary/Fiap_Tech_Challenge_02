import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import  warnings
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

def busca_cotacao_bolsa(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[['Close']]
        return df
    except:
        return pd.DataFrame()
    
def datas_faltantes(df):
    #Criar datas faltantes repetindo o valor ultima data conhecida
    df_close_last = df
    df_close_last = df_close_last.resample('D').ffill()

    return df_close_last

def decomposicao_sazonal(dataframe, modelo, periodo):
    
    df_decompose = seasonal_decompose(dataframe, model=modelo, period=periodo)
    
    return df_decompose

def teste_dickey_fuller(df_close_last):
        
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df_close_last, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    return dfoutput

def wmape(y_true, y_pred):
  return np.abs(y_true-y_pred).sum() / np.abs(y_true).sum()

def rolmean_rolstd(df, window):

    #Rolling statistics - Técnica Visual
    rolmean = df.rolling(window=window).mean().dropna()
    rolstd = df.rolling(window=window).std().dropna()

    return rolmean, rolstd

def transformacao_locaritmica(df):
    
    df_log = np.log(df)
    return df_log

def media_movel(df, window):
    
    df_log = np.log(df)
    moving_avg = df_log.rolling(window=window).mean()
    
    return moving_avg

def subtracao_MV_STL(ts_log, moving_avg):
    
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.dropna(inplace=True)

    return ts_log_moving_avg_diff

def calcula_derivada(ts_log_moving_avg_diff):
    
    df_diff = ts_log_moving_avg_diff.diff(1)
    df_diff.dropna(inplace=True)
    
    return

def renomeia_colunas(df):
    
    #renomear o campo date para ds, campo close para y e criar o campo unique_id
    df_fechamento = df.copy()
    df_fechamento['id'] = range(1, len(df_fechamento) + 1)
    df_fechamento.reset_index(inplace=True)
    df_fechamento = df_fechamento.set_index('id')
    df_fechamento.columns = ['ds', 'y']
    df_fechamento['unique_id'] = 'IBOV'

    return df_fechamento
    
def split_treino_teste(df_modelo, treino_dias, teste_dias):

    treino = df_modelo[-(treino_dias + teste_dias):-teste_dias]
    teste = df_modelo[-teste_dias:]
    h = len(teste.index)
    
    return treino, teste, h

def calcular_e_exibir_diferenca(forecast_df):
    # Seleciona as colunas relevantes e renomeia para nomes mais descritivos
    dif = forecast_df[['ds', 'y', 'yhat']].rename(columns={'ds': 'Data', 'y': 'Valor Real', 'yhat': 'Valor Previsto'})
    
    # Calcula a diferença percentual absoluta entre os valores reais e previstos
    dif['Diferença Percentual (%)'] = np.abs((1 - (dif['Valor Real'] / dif['Valor Previsto']))) * 100
    
    return dif

def hyperparametros_prophet(treino_prophet):
    
    # Definir o espaço de hiperparâmetros
    param_grid = { 
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    # Variáveis para armazenar os resultados
    best_params = None
    best_mae = float('inf')

    # Loop para Grid Search
    for cps in param_grid['changepoint_prior_scale']:
        for sps in param_grid['seasonality_prior_scale']:
            
            # Configurar o modelo com um conjunto de hiperparâmetros
            m = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
            m.fit(treino_prophet)
            
            # Cross-validation
            df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='180 days')
            
            # Calcular métricas de desempenho
            df_p = performance_metrics(df_cv)
            mae = df_p['mae'].mean()
            
            # Atualizar os melhores parâmetros, se necessário
            if mae < best_mae:
                best_mae = mae
                best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps}
   
    return best_params, best_mae

def exibir_periodo_datas(treino, teste):
    # Obtendo as datas mínima e máxima do conjunto de treino
    inicio_treino = treino['ds'].min()
    fim_treino = treino['ds'].max()
    
    # Obtendo as datas mínima e máxima do conjunto de teste
    inicio_teste = teste['ds'].min()
    fim_teste = teste['ds'].max()
    
    return inicio_treino, fim_treino, inicio_teste, fim_teste
