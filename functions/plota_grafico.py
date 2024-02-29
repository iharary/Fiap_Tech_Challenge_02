import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cotacao_ibovespa(df):
    
    # Plota o ibovespa
    # Criar figura e eixo
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plotar os dados de fechamento
    ax.plot(df.index, df['Close'], label='Fechamento', color='blue')

    # Personalizar o gráfico
    ax.set_title("Análise Gráfica do Índice IBOVESPA", fontsize=14)
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Valor de Fechamento", fontsize=12)
    ax.legend()
    plt.xticks(rotation=45) # Rotacionar as datas para melhor visualização

    # Mostrar o gráfico
    plt.tight_layout() # Ajusta automaticamente os parâmetros do subplot para dar espaço ao redor.
    plt.show()

    # Usa st.pyplot() para renderizar a figura no Streamlit
    st.pyplot(fig)

def plot_seasonal_decompose(result):
    
    # Plota o resultado da decomposição
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(20,15))
    result.observed.plot(ax=ax1)
    result.trend.plot(ax=ax2)
    result.seasonal.plot(ax=ax3)
    result.resid.plot(ax=ax4)
    plt.tight_layout()

    # Usa st.pyplot() para renderizar a figura no Streamlit
    st.pyplot(fig)
    
def plot_rolling_statistics(rolmean, rolstd):
    
    # Plota as estatísticas de rolagem
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plotar os dados de fechamento
    ax.plot(rolmean, label='Média Móvel', color='blue')
    ax.plot(rolstd, label='Desvio Padrão Móvel', color='red')

    # Personalizar o gráfico
    ax.set_title("Rolling Statistics", fontsize=14)
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Valor de Fechamento", fontsize=12)
    ax.legend()
    plt.xticks(rotation=45) # Rotacionar as datas para melhor visualização

    # Mostrar o gráfico
    plt.tight_layout() # Ajusta automaticamente os parâmetros do subplot para dar espaço ao redor.
    plt.show()

    # Usa st.pyplot() para renderizar a figura no Streamlit
    st.pyplot(fig)
    
def plot_transformacao_logaritmica(df_log):
    
    # Plota a transformação logaritmica
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plotar os dados de fechamento
    ax.plot(df_log, label='Transformação Logarítmica', color='blue')

    # Personalizar o gráfico
    ax.set_title("Transformação Logarítmica", fontsize=14)
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Valor de Fechamento", fontsize=12)
    ax.legend()
    plt.xticks(rotation=45) # Rotacionar as datas para melhor visualização

    # Mostrar o gráfico
    plt.tight_layout() # Ajusta automaticamente os parâmetros do subplot para dar espaço ao redor.
    plt.show()

    # Usa st.pyplot() para renderizar a figura no Streamlit
    st.pyplot(fig)
    
def plot_media_movel(moving_avg, ts_log):
    # Plota a média móvel e outro dataframe no mesmo gráfico
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plotar os dados de fechamento da média móvel
    ax.plot(moving_avg, label='Média Móvel', color='red')
    
    # Plotar o outro dataframe
    ax.plot(ts_log, label='Transformação Logaritmica', color='blue', linewidth=0.5)

    # Personalizar o gráfico
    ax.set_title("Comparação da Média Móvel com Transformação Logaritmica", fontsize=14)
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.legend()
    plt.xticks(rotation=45) # Rotacionar as datas para melhor visualização

    # Mostrar o gráfico
    plt.tight_layout() # Ajusta automaticamente os parâmetros do subplot para dar espaço ao redor.
    plt.show()

    # Usa st.pyplot() para renderizar a figura no Streamlit
    st.pyplot(fig)
    
def plot_subtracao_MV_STL(ts_log_moving_avg_diff, rolmean, rolstd):
          
    # Inicializa uma figura com dimensões específicas
    fig, ax = plt.subplots(figsize=(15, 10))
    # Plota a série original após a subtração da média móvel
    ax.plot(ts_log_moving_avg_diff, color='blue', label='Original')
    # Plota a média móvel com uma linha mais fina
    ax.plot(rolmean, color='red', linewidth=1, label='Rolling Mean')  # Ajuste a espessura com `linewidth`
    # Plota o desvio padrão
    ax.plot(rolstd, color='black', label='Desvio Padrão Rolante')
    # Adiciona uma legenda no melhor local
    ax.legend(loc='best')
    # Define o título do gráfico
    ax.set_title('Rolling Mean & Rolling Std')
    # Utiliza st.pyplot() para renderizar a figura no Streamlit
    st.pyplot(fig)

def plot_forecast(model, treino, forecast_df):
    
    fig = model.plot(treino, forecast_df, level=[90], unique_ids=['IBOV'],engine ='matplotlib', max_insample_length=90)
    st.pyplot(fig) 
    
def plot_prophet_1(model, forecast):
    
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

def plot_prophet_2(model, forecast):
    
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
    

def plot_prophet_3(forecast_df, treino_prophet):
    
    # Cria a figura e os eixos com dimensões específicas
    fig, ax = plt.subplots(figsize=(15, 10))

    # Utiliza os eixos criados para os gráficos de linha com seaborn
    sns.lineplot(
        data=treino_prophet,
        x='ds',
        y='y',
        label='Treino',
        ax=ax  # Especifica o uso do eixo criado
    )
    sns.lineplot(
        data=forecast_df,
        x='ds',
        y='y',
        label='Real',
        ax=ax  # Especifica o uso do eixo criado
    )
    sns.lineplot(
        data=forecast_df,
        x='ds',
        y='yhat',
        label='Previsão',
        ax=ax  # Especifica o uso do eixo criado
    )

    # Define os nomes dos eixos diretamente através do ax
    ax.set_xlabel('Data')
    ax.set_ylabel('Índice Ibovespa')

    # Exibe a figura no Streamlit
    st.pyplot(fig)
    
